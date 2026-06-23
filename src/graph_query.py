"""
graph_query.py
==============
Multi-hop Query Engine cho GraphRAG.

Quy trình:
1. Nhận câu hỏi → Extract entity chính bằng regex + Groq fallback
2. Tìm node khớp trong graph (fuzzy matching)
3. BFS duyệt 2-hop neighbors → thu thập sub-graph context
4. Textualize: convert triples thành ngôn ngữ tự nhiên
5. Gửi context + câu hỏi lên Groq LLM → lấy câu trả lời
6. Return: {answer, supporting_triples, reasoning_path, tokens_used, time}
"""

import time
from typing import List, Dict, Any, Optional, Set, Tuple

import networkx as nx
from groq import Groq
from rapidfuzz import fuzz, process

# ─────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────
MODEL_NAME      = "llama-3.3-70b-versatile"
MAX_CONTEXT_LEN = 4000   # ký tự tối đa gửi cho LLM
HOP_DEPTH       = 2      # độ sâu BFS
TOP_K_ENTITIES  = 3      # số entity chính cần extract từ câu hỏi
FUZZY_MATCH_THRESHOLD = 65  # ngưỡng fuzzy match để tìm node


ANSWER_PROMPT = """\
You are an expert analyst of the US Electric Vehicle (EV) industry.
You have been given a set of knowledge graph facts (triples) extracted from industry reports.

KNOWLEDGE GRAPH CONTEXT:
{context}

QUESTION: {question}

Instructions:
- Answer based ONLY on the provided context.
- Be specific and cite relevant entities, metrics, and relationships.
- If the context doesn't contain enough information, say so clearly.
- Keep your answer concise but complete (2-5 sentences).

ANSWER:"""


ENTITY_EXTRACT_PROMPT = """\
Extract the key named entities (companies, people, products, technologies, locations) from this question.
Return ONLY a JSON array of strings. No explanation.

Question: {question}

Example output: ["Tesla", "BYD", "United States"]

JSON array:"""


# ─────────────────────────────────────────────────────────────
# Entity extraction từ câu hỏi
# ─────────────────────────────────────────────────────────────
def _extract_query_entities_llm(client: Groq, question: str) -> List[str]:
    """Dùng Groq để extract entities từ câu hỏi."""
    import json, re
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": ENTITY_EXTRACT_PROMPT.format(question=question)}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Làm sạch markdown fence nếu có
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if e]
    except Exception:
        pass
    return []


def _extract_query_entities_regex(question: str) -> List[str]:
    """Heuristic: lấy capitalized words/phrases từ câu hỏi."""
    import re
    # Lấy cụm từ viết hoa (likely named entities)
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b'
    candidates = re.findall(pattern, question)
    # Lọc stop words
    stop_words = {"The", "What", "Who", "When", "Where", "How", "Did", "Does",
                  "Is", "Are", "Was", "Were", "Which", "Why", "Has", "Have",
                  "Do", "Can", "Could", "Would", "Will", "US", "EV", "Q1", "Q2"}
    return [c for c in candidates if c not in stop_words and len(c) > 2]


# ─────────────────────────────────────────────────────────────
# Node matching
# ─────────────────────────────────────────────────────────────
def _find_matching_nodes(G: nx.DiGraph, entity: str, top_k: int = 3) -> List[str]:
    """Tìm node trong graph khớp với entity query bằng fuzzy matching."""
    all_nodes = list(G.nodes())
    if not all_nodes:
        return []

    # Exact match trước
    entity_lower = entity.lower()
    exact = [n for n in all_nodes if entity_lower == n.lower()]
    if exact:
        return exact[:top_k]

    # Fuzzy match
    results = process.extract(
        entity,
        all_nodes,
        scorer=fuzz.token_sort_ratio,
        limit=top_k,
    )
    return [r[0] for r in results if r[1] >= FUZZY_MATCH_THRESHOLD]


# ─────────────────────────────────────────────────────────────
# BFS Traversal (Multi-hop)
# ─────────────────────────────────────────────────────────────
def _bfs_subgraph(
    G: nx.DiGraph,
    start_nodes: List[str],
    hops: int = HOP_DEPTH,
) -> Tuple[Set[str], List[Dict]]:
    """
    BFS từ start_nodes trong phạm vi `hops` hop.
    Returns: (visited_nodes, triples_list)
    """
    visited = set(start_nodes)
    frontier = set(start_nodes)
    all_triples = []
    reasoning_path = list(start_nodes)

    for hop in range(hops):
        next_frontier = set()
        for node in frontier:
            # Duyệt successors (outgoing edges)
            for neighbor in G.successors(node):
                edge_data = G.edges[node, neighbor]
                all_triples.append({
                    "subject": node,
                    "relation": edge_data.get("relation", "RELATED_TO"),
                    "object": neighbor,
                    "weight": edge_data.get("weight", 1),
                })
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

            # Duyệt predecessors (incoming edges)
            for neighbor in G.predecessors(node):
                edge_data = G.edges[neighbor, node]
                all_triples.append({
                    "subject": neighbor,
                    "relation": edge_data.get("relation", "RELATED_TO"),
                    "object": node,
                    "weight": edge_data.get("weight", 1),
                })
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

        frontier = next_frontier
        if not frontier:
            break

    return visited, all_triples


# ─────────────────────────────────────────────────────────────
# Textualization
# ─────────────────────────────────────────────────────────────
def _textualize_triples(triples: List[Dict], max_chars: int = MAX_CONTEXT_LEN) -> str:
    """
    Chuyển triples thành đoạn văn bản có cấu trúc để gửi cho LLM.
    Sort by weight (priority) và cắt ngắn nếu quá dài.
    """
    # Loại bỏ duplicates
    seen = set()
    unique = []
    for t in triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # Sort by weight descending
    unique.sort(key=lambda x: x.get("weight", 1), reverse=True)

    # Textualize
    lines = []
    for t in unique:
        line = f"- {t['subject']} --[{t['relation']}]--> {t['object']}"
        lines.append(line)

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    return text


# ─────────────────────────────────────────────────────────────
# Main Query Function
# ─────────────────────────────────────────────────────────────
def query_graphrag(
    question: str,
    G: nx.DiGraph,
    client: Groq,
    hops: int = HOP_DEPTH,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Trả lời câu hỏi bằng GraphRAG (multi-hop graph traversal + LLM).

    Returns
    -------
    {
        "question": str,
        "answer": str,
        "method": "GraphRAG",
        "supporting_triples": List[Dict],
        "matched_nodes": List[str],
        "context_length": int,
        "time_sec": float,
    }
    """
    start = time.time()

    # ── Step 1: Extract query entities ──
    entities_llm = _extract_query_entities_llm(client, question)
    entities_regex = _extract_query_entities_regex(question)
    # Merge, ưu tiên LLM
    query_entities = list(dict.fromkeys(entities_llm + entities_regex))[:TOP_K_ENTITIES]

    if verbose:
        print(f"  Query entities: {query_entities}")

    # ── Step 2: Find matching nodes ──
    matched_nodes = []
    for entity in query_entities:
        matches = _find_matching_nodes(G, entity)
        matched_nodes.extend(matches)
    matched_nodes = list(dict.fromkeys(matched_nodes))  # dedup

    if not matched_nodes:
        # Fallback: tìm theo substring
        q_lower = question.lower()
        matched_nodes = [n for n in G.nodes() if n.lower() in q_lower][:5]

    if verbose:
        print(f"  Matched graph nodes: {matched_nodes}")

    # ── Step 3: BFS traversal ──
    if matched_nodes:
        visited_nodes, supporting_triples = _bfs_subgraph(G, matched_nodes, hops=hops)
    else:
        supporting_triples = []
        visited_nodes = set()

    # ── Step 4: Textualize context ──
    context = _textualize_triples(supporting_triples)

    if not context:
        context = "No relevant knowledge graph facts found for this question."

    if verbose:
        print(f"  Context length: {len(context)} chars, {len(supporting_triples)} triples")

    # ── Step 5: LLM Answer ──
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()

    elapsed = time.time() - start

    return {
        "question": question,
        "answer": answer,
        "method": "GraphRAG",
        "supporting_triples": supporting_triples[:20],  # top 20 để hiển thị
        "matched_nodes": matched_nodes,
        "visited_nodes": len(visited_nodes),
        "context_length": len(context),
        "time_sec": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────
# Batch query
# ─────────────────────────────────────────────────────────────
def batch_query_graphrag(
    questions: List[str],
    G: nx.DiGraph,
    client: Groq,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Chạy nhiều câu hỏi với GraphRAG."""
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] GraphRAG: {q[:60]}...")
        result = query_graphrag(q, G, client, **kwargs)
        results.append(result)
        time.sleep(0.3)
    return results
