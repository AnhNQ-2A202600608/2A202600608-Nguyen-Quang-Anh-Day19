# 🔬 LAB DAY 19: GraphRAG System với EV Corpus

**Sinh viên:** Nguyễn Quang Anh | **Mã SV:** 2A202600608

## 📋 Tổng Quan

Hệ thống **GraphRAG** hoàn chỉnh xây dựng trên **US Electric Vehicle Corpus** (70 tài liệu).

## 🏗️ Cấu Trúc Dự Án

```
day19/
├── graphrag_pipeline.ipynb    ← Main notebook (chạy từ đây!)
├── requirements.txt           ← Dependencies
├── .env.example               ← Template API key
├── src/
│   ├── data_loader.py         ← Load & chunk 70 docs
│   ├── entity_extractor.py    ← Groq LLM extraction
│   ├── graph_builder.py       ← NetworkX graph
│   ├── graph_query.py         ← Multi-hop BFS query engine
│   ├── flat_rag.py            ← ChromaDB baseline
│   └── evaluator.py           ← 20-question benchmark
├── dataset/                   ← 70 EV corpus documents
└── output/                    ← Results (auto-generated)
    ├── triples.json
    ├── knowledge_graph.gpickle
    ├── knowledge_graph.png    ← 📸 Graph visualization
    ├── evaluation_results.csv ← 📊 Comparison table
    ├── evaluation_chart.png   ← 📈 Charts
    └── final_report.json
```

## 🚀 Hướng Dẫn Chạy

### Bước 1: Cài Đặt
```bash
pip install -r requirements.txt
```

### Bước 2: Lấy Groq API Key (Miễn Phí)
- Đăng ký tại: https://console.groq.com
- Copy `.env.example` → `.env` và điền key

```bash
copy .env.example .env
# Mở .env và điền GROQ_API_KEY=gsk_...
```

### Bước 3: Chạy Notebook
```bash
jupyter notebook graphrag_pipeline.ipynb
```

Chạy từng cell theo thứ tự từ trên xuống dưới.

## ⚡ Pipeline

| Bước | Mô Tả | Tool |
|------|--------|------|
| 1. Load Data | Load & chunk 70 EV documents | `data_loader.py` |
| 2. Extract | Entity/Relation extraction | Groq LLama 3.3 70B |
| 3. Build Graph | NetworkX DiGraph | `graph_builder.py` |
| 4. Query | Multi-hop BFS traversal | `graph_query.py` |
| 5. Baseline | Flat RAG với vector search | ChromaDB + `flat_rag.py` |
| 6. Evaluate | So sánh 20 câu hỏi | `evaluator.py` |

## 📦 Dependencies

- **Groq** — LLM provider (free tier, rất nhanh)
- **NetworkX** — Graph library
- **ChromaDB** — Vector store cho Flat RAG
- **sentence-transformers** — Embeddings
- **rapidfuzz** — Fuzzy entity matching

## 📊 Deliverables

- ✅ `graphrag_pipeline.ipynb` — Notebook chạy end-to-end
- ✅ `output/knowledge_graph.png` — Ảnh đồ thị tri thức
- ✅ `output/evaluation_results.csv` — Bảng 20 câu hỏi
- ✅ `output/evaluation_chart.png` — Biểu đồ so sánh
- ✅ Chi phí: **$0.00** (Groq free tier)
