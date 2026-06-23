"""
entity_extractor.py
===================
Extract entities and relations from text using Groq LLM.
Results are cached to output/triples.json to avoid redundant API calls.

Optimized for Groq free tier rate limits:
- Uses llama-3.1-8b-instant (higher TPD limit)
- Batch-processes multiple chunks per API call
- Saves partial results for resume on failure
- Exponential backoff on 429 rate limit errors
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from groq import Groq
from tqdm import tqdm

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
TRIPLES_FILE = OUTPUT_DIR / "triples.json"
PARTIAL_FILE = OUTPUT_DIR / "triples_partial.json"

MODEL_NAME   = "llama-3.1-8b-instant"
MODEL_LIST   = ["llama-3.1-8b-instant", "qwen/qwen3.6-27b"]
MAX_RETRIES  = 5
RETRY_DELAY  = 3
RATE_LIMIT_WAIT = 30


# Shorter, more efficient prompt to save tokens
EXTRACTION_PROMPT = """\
Extract knowledge graph triples from this text about the EV industry.
Return ONLY a JSON array. Each triple: {{"subject":"...","relation":"UPPER_SNAKE_CASE","object":"..."}}.
Use consistent names (Tesla not TESLA). Extract 5-15 key factual triples.

TEXT:
{text}

JSON:"""


# ---------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------
def _extract_from_chunk(client: Groq, chunk_text: str, doc_id: str, model_name: str) -> List[Dict]:
    """Call Groq API to extract triples from a single chunk."""
    # Trim text aggressively to save tokens
    text = chunk_text[:2000]
    prompt = EXTRACTION_PROMPT.format(text=text)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()

            # Clean thinking tags from reasoning models (e.g. Qwen)
            if "<think>" in raw:
                raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

            # Clean markdown fences
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            # Try to find JSON array in response
            if not raw.startswith("["):
                match = re.search(r"\[[\s\S]*\]", raw)
                if match:
                    raw = match.group(0)

            triples = json.loads(raw)
            if not isinstance(triples, list):
                raise ValueError("Not a JSON array")

            for t in triples:
                t["source_doc"] = doc_id

            return triples

        except Exception as e:
            error_str = str(e)

            # Handle rate limit (429)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = RATE_LIMIT_WAIT
                # Parse wait time from error message
                match = re.search(r"try again in (\d+)m([\d.]+)s", error_str)
                if match:
                    wait_time = int(match.group(1)) * 60 + int(float(match.group(2))) + 5
                else:
                    match = re.search(r"try again in ([\d.]+)s", error_str)
                    if match:
                        wait_time = int(float(match.group(1))) + 5

                print(f"\n  [RATE LIMIT] Waiting {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                continue

            # Other errors: exponential backoff
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                print(f"  [WARN] {doc_id} failed: {error_str[:80]}")
                return []

    return []


def _save_partial(triples: List[Dict], processed_count: int, total_count: int) -> None:
    """Save partial results so progress is not lost."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = {
        "processed_chunks": processed_count,
        "total_chunks": total_count,
        "triples": triples,
    }
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_partial() -> tuple:
    """Load partial results if they exist."""
    if PARTIAL_FILE.exists():
        try:
            with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            triples = data.get("triples", [])
            start = data.get("processed_chunks", 0)
            print(f"[Extractor] Resuming from partial: {start} chunks, {len(triples)} triples")
            return triples, start
        except Exception:
            pass
    return [], 0


def extract_all_triples(
    chunks: List[Dict],
    groq_api_key: str,
    use_cache: bool = True,
    max_chunks: int | None = None,
) -> List[Dict]:
    """
    Extract triples from all chunks.

    Parameters
    ----------
    chunks       : output of data_loader.load_chunks()
    groq_api_key : GROQ_API_KEY
    use_cache    : If True, load from cache if available
    max_chunks   : Limit number of chunks (None = all)

    Returns
    -------
    List of dicts: {subject, relation, object, source_doc}
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- Load from complete cache --
    if use_cache and TRIPLES_FILE.exists():
        print(f"[Extractor] Cache found at {TRIPLES_FILE}")
        with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[Extractor] Loaded {len(cached)} triples from cache.")
        return cached

    # -- Check for partial results to resume --
    all_triples, start_index = _load_partial()

    # -- Extract --
    client = Groq(api_key=groq_api_key)

    if max_chunks:
        chunks = chunks[:max_chunks]

    total_tokens = 0
    start_time = time.time()

    remaining_chunks = chunks[start_index:]
    print(f"[Extractor] Extracting {len(remaining_chunks)} chunks "
          f"(starting from #{start_index}) rotating models {MODEL_LIST}...")

    for i, chunk in enumerate(tqdm(remaining_chunks, desc="Extracting", unit="chunk")):
        model_name = MODEL_LIST[i % len(MODEL_LIST)]
        triples = _extract_from_chunk(client, chunk["text"], chunk["doc_id"], model_name)
        all_triples.extend(triples)
        total_tokens += len(chunk["text"].split()) * 1.3

        # Save partial every 5 chunks
        current_idx = start_index + i + 1
        if current_idx % 5 == 0:
            _save_partial(all_triples, current_idx, len(chunks))

        # Small delay to avoid RPM limit
        time.sleep(0.15)

    elapsed = time.time() - start_time

    # -- Deduplication --
    all_triples = _deduplicate_triples(all_triples)

    # -- Save final cache --
    with open(TRIPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)

    # Clean up partial file
    if PARTIAL_FILE.exists():
        PARTIAL_FILE.unlink()

    print(f"\n[Extractor] Extraction complete!")
    print(f"  Triples: {len(all_triples)}")
    print(f"  Estimated tokens: ~{int(total_tokens):,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved to: {TRIPLES_FILE}")

    return all_triples


def _deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    """Remove exact duplicate triples."""
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


# ---------------------------------------------------------------
# Triple statistics
# ---------------------------------------------------------------
def get_triple_stats(triples: List[Dict]) -> Dict[str, Any]:
    """Return basic statistics about the triples set."""
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
