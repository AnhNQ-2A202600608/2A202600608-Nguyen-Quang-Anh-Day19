"""
flat_rag.py
===========
Baseline Flat RAG sử dụng ChromaDB (vector store) + sentence-transformers.

Pipeline:
1. Index: embed tất cả chunks → lưu vào ChromaDB in-memory collection
2. Query: embed câu hỏi → top-k chunks → Groq LLM answer
"""

import time
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ─────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────
MODEL_NAME       = "openai/gpt-oss-20b"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # nhanh, nhẹ, miễn phí
COLLECTION_NAME  = "ev_corpus"
TOP_K            = 5
MAX_CONTEXT_CHARS = 4000


FLAT_RAG_PROMPT = """\
You are an expert analyst of the US Electric Vehicle (EV) industry.
Based on the following retrieved document excerpts, answer the question.

RETRIEVED DOCUMENTS:
{context}

QUESTION: {question}

Instructions:
- Answer based ONLY on the provided documents.
- Be specific and include relevant data/metrics where available.
- If the documents don't contain enough information, say so clearly.
- Keep your answer concise but complete (2-5 sentences).

ANSWER:"""


# ─────────────────────────────────────────────────────────────
# FlatRAG class
# ─────────────────────────────────────────────────────────────
class FlatRAG:
    """
    Flat RAG baseline sử dụng ChromaDB + sentence-transformers.
    """

    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self._chroma_client = None
        self._collection = None
        self._indexed = False
        print(f"[FlatRAG] Initialized with embedding model: {EMBEDDING_MODEL}")

    def _get_embedding_fn(self):
        """Lazy load sentence-transformers embedding function."""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

    def index(self, chunks: List[Dict], batch_size: int = 100) -> None:
        """
        Index tất cả chunks vào ChromaDB.

        Parameters
        ----------
        chunks : List[{chunk_id, doc_id, title, text}]
        """
        print(f"[FlatRAG] Indexing {len(chunks)} chunks...")
        start = time.time()

        self._chroma_client = chromadb.Client()
        ef = self._get_embedding_fn()

        # Xóa collection cũ nếu tồn tại
        try:
            self._chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Batch add để tránh memory overflow
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in batch],
            )
            if (i // batch_size) % 5 == 0:
                print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")

        elapsed = time.time() - start
        self._indexed = True
        print(f"[FlatRAG] Indexing complete in {elapsed:.1f}s")

    def query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Trả lời câu hỏi bằng Flat RAG.

        Returns
        -------
        {
            "question": str,
            "answer": str,
            "method": "FlatRAG",
            "retrieved_chunks": List[Dict],
            "context_length": int,
            "time_sec": float,
        }
        """
        if not self._indexed:
            raise RuntimeError("Chưa index. Hãy gọi FlatRAG.index(chunks) trước.")

        start = time.time()

        # ── Retrieve top-k chunks ──
        results = self._collection.query(
            query_texts=[question],
            n_results=top_k,
        )

        retrieved_chunks = []
        context_parts = []

        for i, (doc_text, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            chunk_info = {
                "rank": i + 1,
                "text": doc_text,
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "distance": round(float(dist), 4),
            }
            retrieved_chunks.append(chunk_info)

            # Format cho context
            context_parts.append(
                f"[Doc {i+1}: {meta.get('doc_id', '')} - {meta.get('title', '')[:60]}]\n{doc_text}"
            )

        context = "\n\n".join(context_parts)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n... [truncated]"

        # ── LLM Answer ──
        prompt = FLAT_RAG_PROMPT.format(context=context, question=question)
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        import re
        if "<think>" in raw:
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        answer = raw

        elapsed = time.time() - start

        return {
            "question": question,
            "answer": answer,
            "method": "FlatRAG",
            "retrieved_chunks": retrieved_chunks,
            "context_length": len(context),
            "time_sec": round(elapsed, 2),
        }

    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Chạy nhiều câu hỏi với Flat RAG."""
        results = []
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] FlatRAG: {q[:60]}...")
            result = self.query(q, **kwargs)
            results.append(result)
            time.sleep(0.3)
        return results
