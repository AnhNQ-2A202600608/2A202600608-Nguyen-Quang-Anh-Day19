"""
data_loader.py
==============
Load và tiền xử lý 70 tài liệu EV corpus từ thư mục dataset/.
Cung cấp hàm chunk_text để chia nhỏ văn bản.
"""

import os
import re
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "dataset"
CHUNK_SIZE   = 600   # tokens ~ words
CHUNK_OVERLAP = 80


# ─────────────────────────────────────────────────────────────
# Hàm tiện ích
# ─────────────────────────────────────────────────────────────
_NOISE_PATTERNS = re.compile(
    r"(We use cookies|Contact Us|Join our mailing list|Download\s*$|Subscribe|"
    r"Find out more\.|Essential cookies|Google Analytics|privacy policy|"
    r"Terms and conditions|©\d{4}|All Rights Reserved|Fax:|Tel:)",
    re.IGNORECASE,
)

def _clean_text(text: str) -> str:
    """Loại bỏ noise: cookie notices, nav menus, metadata thừa."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if _NOISE_PATTERNS.search(line):
            continue
        if len(line) < 30 and not line.endswith("."):
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def _simple_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chia văn bản thành các chunk theo số từ (word-level), có overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:  # bỏ chunk quá ngắn
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────
# API chính
# ─────────────────────────────────────────────────────────────
def load_documents(dataset_dir: str | Path = DATASET_DIR) -> List[Dict]:
    """
    Load tất cả tài liệu từ dataset_dir.

    Returns
    -------
    List of dicts:
        {
            "doc_id": "doc_1",
            "title": "...",
            "query": "...",
            "url": "...",
            "raw_text": "...",
            "clean_text": "...",
        }
    """
    dataset_dir = Path(dataset_dir)
    docs = []

    txt_files = sorted(dataset_dir.glob("doc_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"[DataLoader] Found {len(txt_files)} documents in {dataset_dir}")

    for txt_file in txt_files:
        raw = txt_file.read_text(encoding="utf-8", errors="ignore")
        lines = raw.splitlines()

        query = ""
        title = ""
        url = ""
        content_start = 0

        for i, line in enumerate(lines):
            line_s = line.strip()
            if line_s.startswith("Query:"):
                query = line_s.removeprefix("Query:").strip()
            elif line_s.startswith("Title:"):
                title = line_s.removeprefix("Title:").strip()
            elif line_s.startswith("Link:"):
                url = line_s.removeprefix("Link:").strip()
            elif line_s == "Full Content:":
                content_start = i + 1
                break

        full_content = "\n".join(lines[content_start:])
        clean = _clean_text(full_content)

        docs.append(
            {
                "doc_id": txt_file.stem,
                "title": title,
                "query": query,
                "url": url,
                "raw_text": full_content,
                "clean_text": clean,
            }
        )

    print(f"[DataLoader] Loaded {len(docs)} documents. "
          f"Total ~{sum(len(d['clean_text'].split()) for d in docs):,} words")
    return docs


def load_chunks(docs: List[Dict] | None = None, **kwargs) -> List[Dict]:
    """
    Chia tất cả tài liệu thành chunks.

    Returns
    -------
    List of dicts:
        {
            "chunk_id": "doc_1_0",
            "doc_id": "doc_1",
            "title": "...",
            "text": "...",
        }
    """
    if docs is None:
        docs = load_documents()

    all_chunks = []
    for doc in docs:
        chunks = _simple_chunk(doc["clean_text"], **kwargs)
        for i, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_{i}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": ch,
                }
            )

    print(f"[DataLoader] Total {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_documents()
    print("\nVí dụ tài liệu đầu tiên:")
    print(f"  ID: {docs[0]['doc_id']}")
    print(f"  Title: {docs[0]['title']}")
    print(f"  Clean text (500 chars): {docs[0]['clean_text'][:500]}")

    chunks = load_chunks(docs)
    print(f"\nVí dụ chunk đầu tiên: {chunks[0]['text'][:300]}")
