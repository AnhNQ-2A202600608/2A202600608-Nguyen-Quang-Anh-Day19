# Task List - Day 19 GraphRAG Lab

- [x] Explore dataset (70 EV docs)
- [x] Create implementation plan
- [x] Clarify requirements (Groq, all 70 docs, ipynb + py)
- [x] Create project structure
- [x] `requirements.txt`
- [x] `src/data_loader.py`
- [x] `src/entity_extractor.py` (Groq API)
- [x] `src/graph_builder.py` (NetworkX)
- [x] `src/graph_query.py` (Multi-hop BFS)
- [x] `src/flat_rag.py` (ChromaDB baseline)
- [x] `src/evaluator.py` (20-question benchmark)
- [x] `graphrag_pipeline.ipynb` (Main notebook)
- [x] `.env.example`, `.gitignore`, `README.md`
- [x] Install dependencies
- [x] Commit initial codebase to GitHub
- [/] Run full end-to-end pipeline test on all 521 chunks (PAUSED: 325/521 chunks completed, 4,628 triples saved in output/triples_partial.json)
- [ ] Verify output files (knowledge_graph.png, subgraphs, evaluation_results.csv, evaluation_chart.png, final_report.json)
- [ ] Push final output artifacts and remaining updates to GitHub
- [ ] Create walkthrough.md summary

---

### How to Resume:
1. Make sure `.env` contains your correct `GROQ_API_KEY`.
2. Run `python run_full_pipeline.py` in the root workspace.
3. The script will automatically detect the cache at `output/triples_partial.json` and resume entity extraction from chunk 325.
4. Models used for rotation: `llama-3.1-8b-instant` and `qwen/qwen3.6-27b` (configured in `src/entity_extractor.py`).
5. After extraction completes, the pipeline will automatically generate the Knowledge Graph, run Flat RAG, evaluate 20 questions, and generate final reports.
