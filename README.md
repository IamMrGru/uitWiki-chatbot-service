# This is the repo for the Final Thesis of Hien and Hai 😎

## Triển khai mô hình

### Chạy requirements.txt

_pip install -r requirements.txt_

### Thêm file .env với Google API key được cung cấp

### Chạy lệnh để deploy local

_streamlit run RAG.py_

### Có thể thực hiện trả lời câu hỏi vì đã có sẵn dữ liệu trong folder faiss_index

## Trong trường hợp muốn điều chỉnh tham số, có thể thực hiện các bước sau:

### Điều chỉnh về các tham số, mô hình

- Điều chỉnh các tham số như chunk_size, chunk_overlap, tham số k trong similarity_search
- Lựa chọn model embeddings, model LLM để sinh câu hỏi và vector store
- Với source code hiện tại,
  - Model Embeddings là: GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
  - Model LLM:GoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.5)
  - Vector DB: FAISS

### Thực hiện add file PDF từ folder metadata_PDF và nhấn nút Process để tạo ra dữ liệu mới trong folder faiss_index
