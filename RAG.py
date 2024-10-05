import streamlit as st
from project_modules.ChainProcess import chain_process
from project_modules.EmbeddingsModel import Embeddingsmodel
from project_modules.LlmModel import llmModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

def filter_by_metadata(question,new_db):
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="Tên của quy định",
            type="string",
        ),
        AttributeInfo(
            name="tags", 
            description="Các từ khóa hoặc cụm từ mô tả nội dung của tài liệu, giúp dễ dàng tìm kiếm và phân loại tài liệu theo các chủ đề cụ thể", 
            type="string"
        ),
        AttributeInfo(
            name="target audience", 
            description="Đối tượng cần sử dụng", 
            type="string"
        ),
        ]   
    document_content_description = "Tóm tắt cho nội dung của tài liệu"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)
    retriever = SelfQueryRetriever.from_llm(
    llm,
    new_db,
    document_content_description,
    metadata_field_info,
    verbose=True
    )
    docs=retriever.invoke(question)
    return docs


#---------------------------------------------
# Hàm xử lý câu hỏi từ người dùng
def user_input(user_question):

    # Chọn embeddings để tạo vector store
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Phải lựa chọn đúng model đã sử dụng để tạo vector store
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") text-multilingual-embedding-002
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = Embeddingsmodel.HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")

    # Load vector store đã lưu với metadata
    # new_db = Embeddingsmodel.FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) 
    # docs = new_db.similarity_search(user_question,k=10) # Lấy 6 chunk có độ tương đồng về nội dung với câu hỏi của người dùng

    #Pinecone
    new_db= Embeddingsmodel.PineconeVectorStore(index_name='testing',embedding=embeddings)
    docs1=filter_by_metadata(user_question,new_db)
    docs2=new_db.similarity_search(query=user_question,k=3) # Lấy 10  chunk có độ tương đồng về nội dung với câu hỏi của người dùng
    docs=docs1

    metadata_combined = "\n".join([str(doc.metadata) for doc in docs]) # Kết hợp metadata của các chunk

    chain = llmModel.get_conversational_chain() # Khởi tạo chain cho việc hỏi đáp
    # Gửi câu hỏi từ người dùng và các chunk có độ tương đồng cho chain để nhận câu trả lời
    response = chain({"input_documents": docs,"metadata": metadata_combined, "question": user_question}, return_only_outputs=True)
    print('Response là: ', response)
    st.write("UITBot: ", response['output_text'])


#---------------------------------------------
# Hàm main để chạy ứng dụng Streamlit
def main():
    st.header("Chat with UITWikiBot")
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
                chunks_with_metadata = chain_process.get_pdf_text_with_metadata(pdf_docs)
                # Tạo vector store với metadata
                Embeddingsmodel.get_vector_store_with_metadata(chunks_with_metadata)
                st.success("Processing complete")

if __name__ == "__main__":
    main()
