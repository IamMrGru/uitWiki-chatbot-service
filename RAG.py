import streamlit as st
from project_module.ChainProcess import pdf_process
from project_module.EmbeddingsModel import Embeddingsmodel
from project_module.LlmModel import llmModel

# Hàm main để chạy ứng dụng Streamlit
def main():
    st.header("Chat with UITWikiBot")
    user_question = st.text_input("Ask a question")
    if user_question:
        llmModel.user_input(user_question)
    chunks_with_metadata=[]
    with st.sidebar:
        st.title("Menu")
        # Upload file PDF
        pdf_docs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)
        if st.button("Submit and process"):
            with st.spinner("Processing"):
                # Xử lý văn bản và metadata từ file PDF
                chunks_with_metadata = pdf_process.get_pdf_text_with_metadata(pdf_docs)

                # Test hiển thị chunks_with_metadata sau khi xử lý
                if chunks_with_metadata:
                    st.title("Text after chunking")
                    st.write(chunks_with_metadata)
                # else:
                #     st.write("No chunks with metadata found")

                # Biến đổi các chunk văn bản thành vector kèm với metadata và lưu vào Vector Database 
                Embeddingsmodel.get_vector_store_with_metadata(chunks_with_metadata)
                st.success("Processing complete")
    
if __name__ == "__main__":
    main()
