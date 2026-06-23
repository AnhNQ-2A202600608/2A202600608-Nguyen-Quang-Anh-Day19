"""
entity_extractor.py
===================
Trích xuất entities và relations từ văn bản bằng Groq LLM.
Kết quả được cache vào output/triples.json để tránh gọi API lại.

Entity types được nhận diện:
  COMPANY   - Tesla, BYD, Ford, OpenAI, ...
  PERSON    - Elon Musk, Sam Altman, ...
  PRODUCT   - Model 3, Lyriq, Seagull, ...
  CONCEPT   - Electric Vehicle, Battery, ZEV Regulation, ...
  LOCATION  - United States, China, Europe, California, ...
  METRIC    - Market share, Price, Growth rate, ...
  DATE/YEAR - 2024, Q1 2023, ...

Relation types (ví dụ):
  FOUNDED_BY, PRODUCES, ACQUIRED, COMPETES_WITH, LOCATED_IN,
  INVESTED_IN, PARTNERED_WITH, REPORTED_REVENUE, HAS_MARKET_SHARE,
  SURPASSED, REGULATION_APPLIES_TO, ...
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from groq import Groq
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
TRIPLES_FILE = OUTPUT_DIR / "triples.json"
MODEL_NAME   = "llama-3.3-70b-versatile"   # Groq free tier, nhanh
MAX_RETRIES  = 3
RETRY_DELAY  = 2  # seconds


EXTRACTION_PROMPT = """\
You are an expert knowledge graph builder specializing in the electric vehicle (EV) industry.

Your task: Extract knowledge graph triples from the given text.

RULES:
1. Extract ONLY factual, specific information — avoid vague claims.
2. Each triple must be: {{"subject": "...", "relation": "...", "object": "..."}}.
3. Use consistent entity names (e.g., always "Tesla", never "Tesla Inc." or "TESLA").
4. Subject and Object must be NAMED ENTITIES (companies, people, products, places, metrics).
5. Relation should be a clear, active verb phrase in UPPER_SNAKE_CASE.
6. Extract 5–25 triples per chunk. Quality over quantity.
7. Return ONLY a valid JSON array. No explanation, no markdown fences.

GOOD EXAMPLE:
[
  {{"subject": "Tesla", "relation": "HAS_MARKET_SHARE", "object": "51.3% in Q1 2024"}},
  {{"subject": "BYD", "relation": "SURPASSED", "object": "Tesla as largest EV producer"}},
  {{"subject": "Cadillac Lyriq", "relation": "ACHIEVED_GROWTH", "object": "499.2% YoY in Q1 2024"}},
  {{"subject": "Cox Automotive", "relation": "ANALYZES", "object": "US EV sales data"}}
]

TEXT TO PROCESS:
{text}

JSON ARRAY OF TRIPLES:"""


# ─────────────────────────────────────────────────────────────
# Core extraction logic
# ─────────────────────────────────────────────────────────────
def _extract_from_chunk(client: Groq, chunk_text: str, doc_id: str) -> List[Dict]:
    """Gọi Groq API để extract triples từ một chunk."""
    prompt = EXTRACTION_PROMPT.format(text=chunk_text[:3000])  # giới hạn độ dài

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            # Làm sạch nếu model trả về markdown fence
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            triples = json.loads(raw)
            if not isinstance(triples, list):
                raise ValueError("Response is not a JSON array")

            # Gắn thêm source doc
            for t in triples:
                t["source_doc"] = doc_id

            return triples

        except (json.JSONDecodeError, ValueError, Exception) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  [WARN] Chunk từ {doc_id} failed sau {MAX_RETRIES} lần: {e}")
                return []

    return []


import re  # đặt ở đây để tránh circular import ở đầu


def extract_all_triples(
    chunks: List[Dict],
    groq_api_key: str,
    use_cache: bool = True,
    max_chunks: int | None = None,
) -> List[Dict]:
    """
    Extract triples từ tất cả chunks.

    Parameters
    ----------
    chunks       : output của data_loader.load_chunks()
    groq_api_key : GROQ_API_KEY
    use_cache    : Nếu True, load từ cache nếu có (bỏ qua API call)
    max_chunks   : Giới hạn số chunks (None = tất cả)

    Returns
    -------
    List of dicts: {subject, relation, object, source_doc}
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load từ cache ──
    if use_cache and TRIPLES_FILE.exists():
        print(f"[Extractor] Cache found at {TRIPLES_FILE}")
        with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[Extractor] Loaded {len(cached)} triples from cache.")
        return cached

    # ── Extract mới ──
    client = Groq(api_key=groq_api_key)

    if max_chunks:
        chunks = chunks[:max_chunks]

    all_triples: List[Dict] = []
    total_tokens = 0
    start_time = time.time()

    print(f"[Extractor] Starting extraction of {len(chunks)} chunks with model {MODEL_NAME}...")

    for chunk in tqdm(chunks, desc="Extracting", unit="chunk"):
        triples = _extract_from_chunk(client, chunk["text"], chunk["doc_id"])
        all_triples.extend(triples)
        total_tokens += len(chunk["text"].split()) * 1.3  # rough estimate
        time.sleep(0.2)  # tránh rate limit

    elapsed = time.time() - start_time

    # ── Deduplication cơ bản ──
    all_triples = _deduplicate_triples(all_triples)

    # ── Lưu cache ──
    with open(TRIPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)

    print(f"\n[Extractor] ✅ Hoàn thành!")
    print(f"  Triples: {len(all_triples)}")
    print(f"  Estimated tokens: ~{int(total_tokens):,}")
    print(f"  Thời gian: {elapsed:.1f}s")
    print(f"  Đã lưu vào: {TRIPLES_FILE}")

    return all_triples


def _deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    """Loại bỏ triples trùng lặp hoàn toàn."""
    seen = set()
    unique = []
    for t in triples:
        key = (
            t.get("subject", "").lower().strip(),
            t.get("relation", "").lower().strip(),
            t.get("object", "").lower().strip(),
        )
        if key not in seen and all(k for k in key):
            seen.add(key)
            unique.append(t)
    print(f"[Extractor] Dedup: {len(triples)} -> {len(unique)} triples")
    return unique


# ─────────────────────────────────────────────────────────────
# Thống kê triples
# ─────────────────────────────────────────────────────────────
def get_triple_stats(triples: List[Dict]) -> Dict[str, Any]:
    """Trả về thống kê cơ bản về tập triples."""
    subjects  = [t["subject"] for t in triples]
    objects   = [t["object"]  for t in triples]
    relations = [t["relation"] for t in triples]

    all_entities = list(set(subjects + objects))

    from collections import Counter
    top_entities  = Counter(subjects + objects).most_common(20)
    top_relations = Counter(relations).most_common(10)
    top_subjects  = Counter(subjects).most_common(10)

    return {
        "total_triples": len(triples),
        "unique_entities": len(all_entities),
        "unique_relations": len(set(relations)),
        "top_entities": top_entities,
        "top_relations": top_relations,
        "top_subjects": top_subjects,
    }
