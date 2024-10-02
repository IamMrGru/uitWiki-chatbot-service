from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#---------------------------------------------
# Hàm để trích xuất văn bản và metadata từ file PDF
def get_pdf_text_with_metadata(pdf_docs):
    text_chunks_with_metadata = []
    
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        
        # Trích xuất metadata
        pdf_metadata = {
            "title": pdf_reader.metadata['/Title'],
            "author": pdf_reader.metadata['/Author'],
            "description": pdf_reader.metadata['/Description'],
            "category": pdf_reader.metadata['/Category'],
            "tags": pdf_reader.metadata['/Tags'],
            "target": pdf_reader.metadata['/Target Audience'],
        }
        
        # Trích xuất nội dung và gắn metadata
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            # Chia văn bản thành các chunk nhỏ hơn
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=90)
            chunks = text_splitter.split_text(page_text)
            
            # Gắn metadata với từng chunk
            for chunk in chunks:
                chunk_with_metadata = {
                    "content": chunk,
                    "metadata": {
                        "page_number": page_num + 1,
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "description": pdf_metadata.get("description", ""),
                        "category": pdf_metadata.get("description", ""),
                        "tags": pdf_metadata.get("tags", ""),
                        "target audience": pdf_metadata.get("target", ""),
                    }
                }
                text_chunks_with_metadata.append(chunk_with_metadata)
    
    return text_chunks_with_metadata