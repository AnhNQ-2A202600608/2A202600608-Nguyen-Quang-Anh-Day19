"""
run_full_pipeline.py
====================
Chay toan bo GraphRAG pipeline end-to-end de test.
Tuong duong voi viec chay tat ca cells trong graphrag_pipeline.ipynb
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ================================================================
# STEP 0: Setup
# ================================================================
print("=" * 70)
print("  GRAPHRAG PIPELINE - FULL TEST RUN")
print("  Lab Day 19: EV Industry Knowledge Graph")
print("=" * 70)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("ERROR: GROQ_API_KEY not found in .env!")
    sys.exit(1)

from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Quick test
try:
    test_resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say OK in one word"}],
        max_tokens=5,
    )
    print(f"\n[STEP 0] Groq API OK: {test_resp.choices[0].message.content.strip()}")
except Exception as e:
    print(f"ERROR: Groq API connection failed: {e}")
    sys.exit(1)

# ================================================================
# STEP 1: Data Loading
# ================================================================
print("\n" + "=" * 70)
print("  STEP 1: Loading & Chunking Documents")
print("=" * 70)

from data_loader import load_documents, load_chunks

docs = load_documents(dataset_dir=Path(__file__).parent / "dataset")
chunks = load_chunks(docs)

total_words = sum(len(d["clean_text"].split()) for d in docs)
print(f"\n  Documents: {len(docs)}")
print(f"  Total words: {total_words:,}")
print(f"  Chunks: {len(chunks)}")
print(f"  Avg words/chunk: {total_words // len(chunks)}")

# ================================================================
# STEP 2: Entity & Relation Extraction
# ================================================================
print("\n" + "=" * 70)
print("  STEP 2: Entity & Relation Extraction (Groq LLM)")
print("  This may take 5-15 minutes on first run (results are cached)")
print("=" * 70)

from entity_extractor import extract_all_triples, get_triple_stats

start_extract = time.time()
triples = extract_all_triples(
    chunks=chunks,
    groq_api_key=GROQ_API_KEY,
    use_cache=True,
    max_chunks=None,
)
extract_time = time.time() - start_extract

stats = get_triple_stats(triples)
print(f"\n  Extraction time: {extract_time:.1f}s")
print(f"  Total triples: {stats['total_triples']:,}")
print(f"  Unique entities: {stats['unique_entities']:,}")
print(f"  Unique relations: {stats['unique_relations']:,}")

print(f"\n  Top 10 entities:")
for entity, count in stats["top_entities"][:10]:
    print(f"    {entity}: {count}")

print(f"\n  Top 10 relations:")
for rel, count in stats["top_relations"][:10]:
    print(f"    {rel}: {count}")

print(f"\n  Sample triples:")
for t in triples[:5]:
    print(f"    ({t['subject']}) --[{t['relation']}]--> ({t['object']})")

# ================================================================
# STEP 3: Knowledge Graph Construction
# ================================================================
print("\n" + "=" * 70)
print("  STEP 3: Building Knowledge Graph (NetworkX)")
print("=" * 70)

from graph_builder import build_graph, visualize_graph, visualize_subgraph, get_graph_stats

G = build_graph(triples)
graph_stats = get_graph_stats(G)

print(f"\n  Entity type distribution:")
for etype, count in sorted(graph_stats["entity_types"].items(), key=lambda x: -x[1]):
    bar = "#" * (count // 3)
    print(f"    {etype:10s}: {count:4d} {bar}")

print(f"\n  Top 10 most connected nodes:")
for node, degree in graph_stats["top_nodes_by_degree"]:
    ntype = G.nodes[node].get("type", "OTHER")
    print(f"    [{ntype:8s}] {node}: degree={degree}")

# Visualize
print("\n  Generating visualizations...")
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

visualize_graph(G, title="EV Industry Knowledge Graph - Top 80 Entities", top_n=80,
                save_path=output_dir / "knowledge_graph.png")

# Try to visualize Tesla and BYD subgraphs
for entity in ["Tesla", "BYD", "China"]:
    if entity in G.nodes():
        visualize_subgraph(G, center_node=entity, hops=2,
                           save_path=output_dir / f"subgraph_{entity}.png")
    else:
        # Try fuzzy match
        matches = [n for n in G.nodes() if entity.lower() in n.lower()]
        if matches:
            visualize_subgraph(G, center_node=matches[0], hops=2,
                               save_path=output_dir / f"subgraph_{entity}.png")

# ================================================================
# STEP 4: GraphRAG Query Demo
# ================================================================
print("\n" + "=" * 70)
print("  STEP 4: GraphRAG Multi-hop Query Demo")
print("=" * 70)

from graph_query import query_graphrag

demo_questions = [
    "What is the relationship between BYD and Berkshire Hathaway?",
    "How did Tesla's price cuts in China affect competition?",
    "What role does CATL play in the global battery supply chain?",
]

for q in demo_questions:
    print(f"\n  Q: {q}")
    result = query_graphrag(q, G, groq_client, verbose=False)
    print(f"  A: {result['answer'][:200]}")
    print(f"     [Nodes: {result['visited_nodes']}, Time: {result['time_sec']}s]")
    time.sleep(0.5)

# ================================================================
# STEP 5: Flat RAG Baseline
# ================================================================
print("\n" + "=" * 70)
print("  STEP 5: Flat RAG Baseline (ChromaDB)")
print("=" * 70)

from flat_rag import FlatRAG

flat_rag = FlatRAG(groq_api_key=GROQ_API_KEY)
flat_rag.index(chunks)

# Quick demo
flat_result = flat_rag.query("What is Tesla's market share in Q1 2024?")
print(f"\n  Demo query: What is Tesla's market share in Q1 2024?")
print(f"  Answer: {flat_result['answer'][:200]}")
print(f"  Retrieved {len(flat_result['retrieved_chunks'])} chunks, Time: {flat_result['time_sec']}s")

# ================================================================
# STEP 6: Full Evaluation (20 Questions)
# ================================================================
print("\n" + "=" * 70)
print("  STEP 6: Running 20-Question Benchmark")
print("  Comparing Flat RAG vs GraphRAG")
print("=" * 70)

from evaluator import BENCHMARK_QUESTIONS, run_evaluation, print_evaluation_summary, plot_evaluation
from graph_query import batch_query_graphrag

questions_list = [q["question"] for q in BENCHMARK_QUESTIONS]

print("\n  Running Flat RAG on 20 questions...")
flat_results = flat_rag.batch_query(questions_list)

print("\n  Running GraphRAG on 20 questions...")
graph_results = batch_query_graphrag(questions_list, G, groq_client)

# Evaluate
df_eval = run_evaluation(flat_results, graph_results)
print_evaluation_summary(df_eval)

# Plot
plot_evaluation(df_eval)

# ================================================================
# STEP 7: Cost Analysis
# ================================================================
print("\n" + "=" * 70)
print("  STEP 7: Cost Analysis")
print("=" * 70)

avg_chunk_words = sum(len(c["text"].split()) for c in chunks) / len(chunks)
est_input = len(chunks) * avg_chunk_words * 1.3
est_output = len(chunks) * 150

total_flat_time = sum(r["time_sec"] for r in flat_results)
total_graph_time = sum(r["time_sec"] for r in graph_results)

print(f"\n  INDEXING COST:")
print(f"    Chunks processed: {len(chunks)}")
print(f"    Est. input tokens: {est_input:,.0f}")
print(f"    Est. output tokens: {est_output:,.0f}")
print(f"    Groq cost: $0.00 (free tier)")
print(f"    OpenAI equivalent: ~${(est_input * 0.15 + est_output * 0.6) / 1e6:.3f}")

print(f"\n  QUERY COST (20 questions):")
print(f"    FlatRAG total: {total_flat_time:.1f}s (avg {total_flat_time/20:.1f}s/q)")
print(f"    GraphRAG total: {total_graph_time:.1f}s (avg {total_graph_time/20:.1f}s/q)")

# ================================================================
# STEP 8: Save Final Report
# ================================================================
print("\n" + "=" * 70)
print("  STEP 8: Saving Final Report")
print("=" * 70)

final_report = {
    "metadata": {
        "student": "Nguyen Quang Anh",
        "student_id": "2A202600608",
        "lab": "Day 19 - GraphRAG",
        "dataset": f"{len(docs)} EV Corpus documents",
        "llm": "Groq llama-3.1-8b-instant",
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "graph_stats": {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_triples": len(triples),
        "entity_types": graph_stats["entity_types"],
    },
    "evaluation": {
        "flat_rag_avg_score": round(df_eval["FlatRAG_Score"].mean(), 2),
        "graphrag_avg_score": round(df_eval["GraphRAG_Score"].mean(), 2),
        "winner_counts": df_eval["Winner"].value_counts().to_dict(),
        "flat_rag_avg_time": round(df_eval["FlatRAG_Time_s"].mean(), 2),
        "graphrag_avg_time": round(df_eval["GraphRAG_Time_s"].mean(), 2),
    },
    "cost": {
        "groq_cost_usd": 0.00,
        "openai_equivalent_usd": round((est_input * 0.15 + est_output * 0.6) / 1e6, 3),
        "total_extraction_time_s": round(extract_time, 1),
    },
}

with open(output_dir / "final_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)

print(f"\n  Saved: output/final_report.json")
print(f"\n  All output files:")
for fpath in sorted(output_dir.iterdir()):
    if fpath.name.startswith("."):
        continue
    size = fpath.stat().st_size
    print(f"    {fpath.name}: {size:,} bytes")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE!")
print("=" * 70)
print(f"""
  Documents:   {len(docs)}
  Chunks:      {len(chunks)}
  Triples:     {len(triples):,}
  Graph Nodes: {G.number_of_nodes():,}
  Graph Edges: {G.number_of_edges():,}

  FlatRAG avg score:  {df_eval['FlatRAG_Score'].mean():.2f}/10
  GraphRAG avg score: {df_eval['GraphRAG_Score'].mean():.2f}/10

  Deliverables:
    [x] Source code (src/*.py)
    [x] Knowledge graph screenshot (output/knowledge_graph.png)
    [x] 20-question benchmark table (output/evaluation_results.csv)
    [x] Comparison chart (output/evaluation_chart.png)
    [x] Cost analysis (output/final_report.json)
""")
