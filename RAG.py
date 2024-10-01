from PyPDF2 import PdfReader
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# GOOGLE API KEY từ env
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


#---------------------------------------------
# Hàm để trích xuất văn bản và metadata từ file PDF
def get_pdf_text_with_metadata(pdf_docs):
    text_chunks_with_metadata = []
    
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        
        # Trích xuất metadata từ file PDF đã được gắn nhãn trước đó
        pdf_metadata = {
            "topic": pdf_reader.metadata['/Topic'],
            "department": pdf_reader.metadata['/Department'],
        }
        
        # Trích xuất nội dung và gắn metadata
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            # Chia văn bản thành các chunk nhỏ hơn
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(page_text)
            
            # Gắn metadata với từng chunk
            for chunk in chunks:
                chunk_with_metadata = {
                    "content": chunk,
                    "metadata": {
                        "page_number": page_num + 1,
                        "topic": pdf_metadata.get("topic", ""),
                        "department": pdf_metadata.get("department", ""),
                    }
                }
                text_chunks_with_metadata.append(chunk_with_metadata)
    
    return text_chunks_with_metadata


#---------------------------------------------
# Hàm để tạo VectorStore với metadata
def get_vector_store_with_metadata(chunks_with_metadata):

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Có thể lựa chọn model Embedding khác
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


    # Chuyển các chunk với metadata thành dạng embedding và lưu vào vector store local
    texts = [chunk["content"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas) # Có thể thay đổi vector store khác
    vector_store.save_local('faiss_index')


#---------------------------------------------
# Hàm khởi tạo chain cho việc hỏi đáp
def get_conversational_chain():
    prompt_template="""
    Vai trò của UITBot:
    - Bạn là một trợ lý ảo giải đáp của sinh viên tại trường đại học UIT.
    - Nhiệm vụ của bạn là trả lời các câu hỏi và thắc mắc của sinh viên một cách chi tiết và chính xác nhất.
    - Metadata chính là điều kiện lọc để bạn có thể chắt lọc thông tin. Context có thể chính là nơi chứa đáp án của câu hỏi.
    - Hãy trả lời câu hỏi dựa trên các thông tin được cung cấp trong context và metadata.
    - Trong đó, context chính là nơi chứa đáp án của câu hỏi và metadata cũng có thể trả lời một số thông tin quan trọng đi kèm
    -------------------------

    Dưới đây là thông tin cần thiết:
    *METADATA* là: ({metadata})
    ---
    *CONTEXT* là: ({context})
    ---
    *QUESTION* là: ({question})
    -------------------------

    Yêu cầu về câu trả lời:
    - Hãy đảm bảo cung cấp đầy đủ chi tiết theo METADATA,CONTEXT.
    - Hãy sắp xếp câu trả lời thành một cấu trúc đẹp. Bạn không cần phải trả lời dựa vào đâu (Dựa vào METADATA được cung cấp...., dựa vào CONTEXT ta thấy,...)
    - Đưa ra một câu trả lời tự nhiên và dễ hiểu nhất có thể.
    - Không tự trả lời mà không có trong METADATA,CONTEXT. Nếu không có hãy trả lời (Vui lòng cung cấp thêm thông tin chi tiết)
    """
    
    #Chọn model để tạo chain
    # model=ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0) # Có thể lựa chọn model LLM khác
    model=GoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.5)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question",'metadata'])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt,verbose=True)
    return chain

#---------------------------------------------
# Hàm xử lý câu hỏi từ người dùng
def user_input(user_question):

    # Chọn embeddings để tạo vector store
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Phải lựa chọn đúng model đã sử dụng để tạo vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load vector store đã lưu với metadata
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question,k=6) # Lấy 6 chunk có độ tương đồng về nội dung với câu hỏi của người dùng
    metadata_combined = "\n".join([str(doc.metadata) for doc in docs]) # Kết hợp metadata của các chunk

    chain = get_conversational_chain() # Khởi tạo chain cho việc hỏi đáp
    # Gửi câu hỏi từ người dùng và các chunk có độ tương đồng cho chain để nhận câu trả lời
    response = chain({"input_documents": docs,"metadata": metadata_combined, "question": user_question}, return_only_outputs=True)
    print('Response là: ', response)
    st.write("UITBot: ", response['output_text'])


#---------------------------------------------
# Hàm main để chạy ứng dụng Streamlit
def main():
    st.header("Chat with my PDF using Gemini")
    user_question = st.text_input("Ask a question")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        # Upload file PDF
        pdf_docs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)
        if st.button("Submit and process"):
            with st.spinner("Processing"):
                # Xử lý văn bản và metadata từ file PDF
                chunks_with_metadata = get_pdf_text_with_metadata(pdf_docs)
                # Tạo vector store với metadata
                get_vector_store_with_metadata(chunks_with_metadata)
                st.success("Processing complete")

if __name__ == "__main__":
    main()
