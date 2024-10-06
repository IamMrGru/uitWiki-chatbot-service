""" # Chứa hàm xử lý chuỗi dữ liệu lấy từ PDF và gắn metadata với từng chunk văn bản """
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


#---------------------------------------------
def get_pdf_text_with_metadata(pdf_docs):

    """ ## Hàm để trích xuất văn bản và metadata từ file PDF và gắn metadata với từng chunk văn bản """

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=516, chunk_overlap=60) # 850
            chunks = text_splitter.split_text(page_text)
            
            # Gắn metadata với từng chunk
            for chunk in chunks:
                category=pdf_metadata.get("category", "")
                tags=pdf_metadata.get("tags", "")
                
                print(chunk)
                # Nhập thông tin metadata tùy ý cho từng chunk
                print('-------------------------------------------------------------')
                print('Bạn có muốn nhập thông tin cho chunk này không? (Y/N)')
                answer=input()
                if answer.lower()=='y':
                    print('Nhập thông tin cho chunk này')
                    print('----------------')
                    category=input("Nhập danh mục: ")
                    tags=input("Nhập tags: ")
                
                
                chunk_with_metadata = {
                    "content": chunk,
                    "metadata": {
                        "page_number": page_num + 1,
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "description": pdf_metadata.get("description", ""),
                        "category": category,
                        "tags": tags,
                        "target audience": pdf_metadata.get("target", ""),
                    }
                }
                text_chunks_with_metadata.append(chunk_with_metadata)
                print('Đã nhập xong thông tin cho chunk này')
                print('-------------------------------------------------------------')
    
    return text_chunks_with_metadata
