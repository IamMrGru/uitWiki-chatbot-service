""" # Sử dụng LLM để generate câu trả lời tự nhiên từ câu hỏi người dùng nhập vàp"""

#LLM Model
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_anthropic import AnthropicLLM
from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from project_modules.EmbeddingsModel import Embeddingsmodel
import streamlit as st
from dotenv import load_dotenv
import os
import time

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Hàm để hiển thị ký tự
def generate_text_slowly(text, delay=0.00001):
    text_placeholder = st.empty()  # Tạo một placeholder để cập nhật nội dung
    displayed_text = ""
    for char in text:
        displayed_text += char
        text_placeholder.markdown(displayed_text)  # Cập nhật nội dung tại placeholder
        time.sleep(delay)  # Tạo độ trễ để hiển thị dần dần

#Hàm để SelfQueryRetriever lọc các metadata
def filter_by_metadata(question,new_db):
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="Tên của tài liệu chứa thông tin cần truy xuất", 
            type="string",
        ),
        AttributeInfo(
            name="author",
            description= "Phòng ban quản lý tài liệu", 
            type="string",
        ),
        AttributeInfo(
            name="description",
            description= "Mô tả nội dung của tài liệu", 
            type="string",
        ),
        AttributeInfo(
            name="category",
            description= "Quy định, chính sách, quy trình, hướng dẫn", 
            type="string",
        ),
        AttributeInfo(
            name="tags", 
            description="Các từ khóa liên quan đến đoạn văn cần truy xuất", 
            type="string"
        ),
        AttributeInfo(
            name="target audience", 
            description="Đối tượng cần sử dụng", 
            type="string"
        ),
        ]   
    document_content_description = "Tóm tắt nội dung của tài liệu trên "
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)
    retriever = SelfQueryRetriever.from_llm(
    llm,
    new_db,
    document_content_description,
    metadata_field_info,
    verbose=True,
    )
    docs=retriever.invoke(question)
    return docs


#---------------------------------------------
def get_conversational_chain():
    """ ## Hàm khởi tạo chain cho việc hỏi đáp"""


    prompt_template="""
    Tên của bạn là UITWikiBot. 
    Được phát triển bởi nhóm sinh viên UIT: Hiển Đoàn và Hải Đào dưới sự hướng dẫn của thầy Tín.
    Vai trò của bạn là:
    - Bạn là một trợ lý ảo giải đáp của sinh viên tại trường Đại học Công nghệ Thông tin UIT.
    - Nhiệm vụ của bạn là trả lời các câu hỏi và thắc mắc của sinh viên một cách chi tiết và chính xác nhất.
    - Metadata cũng có thể là chứa keyword mà trong câu hỏi có. Context có thể chính là nơi chứa đáp án của câu hỏi.
    - Hãy trả lời câu hỏi dựa trên các thông tin được cung cấp trong context và metadata.
    - Trong đó, context chính là nơi chứa đáp án của câu hỏi và metadata cũng có thể trả lời một số thông tin quan trọng đi kèm
    - Context là những mẫu bối cảnh rời rạc. Do đó, bạn hãy chắt lọc, ghép nối các context để trả lời câu hỏi một cách hợp lý .
    -------------------------

    Dưới đây là thông tin tôi sẽ đưa bạn :
    *METADATA* là: ({metadata})
    ---
    *CONTEXT* là: ({context})
    ---
    *QUESTION* là: ({question}?)
    -------------------------

    Yêu cầu về câu trả lời:
    - Hãy đảm bảo cung cấp đầy đủ chi tiết theo METADATA,CONTEXT.
    - Cố gắng liên kết thông tin giữa METADATA,CONTEXT để tạo ra câu trả lời chính xác nhất.
    - Hãy sắp xếp câu trả lời thành một cấu trúc đẹp. Ở những câu trả lời về quy định, các bước thực hiện, hãy sắp xếp câu trả lời theo thứ tự
    Bạn không cần phải trả lời dựa vào đâu (Dựa vào METADATA được cung cấp...., dựa vào CONTEXT ta thấy,...)
    - Đưa ra một câu trả lời tự nhiên và dễ hiểu nhất có thể.
    - Không tự trả lời mà không có trong METADATA,CONTEXT. Nếu không có hãy trả lời (Vui lòng cung cấp thêm thông tin chi tiết)
    - Bạn có thể trích dẫn tên tài liệu chứa thông tin hoặc nội dung đó nằm ở phần nào của tài liệu đó.
    - Với những dạng YES/NO, hãy trả lời rõ ràng và chi tiết nhất có thể, phải giải thích vì sao trả lời như vậy dựa trên trích dẫn thông tin đó lấy từ tài liệu nào 
    - Những câu trả lời có đường dẫn đến link URL hay đường dẫn để download, bạn hãy embed link đó vào câu trả lời của mình .
    - Hãy embed đường dẫn tải các mẫu đơn vào tên mẫu đơn đó.
       - Ví dụ như : [Đường dẫn tải mẫu đơn](https://www.uit.edu.vn)

    """
    
    #Chọn model để tạo chain
    #model=ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0) # Có thể lựa chọn model LLM khác
    model=GoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)
    # model=HuggingFaceEndpoint(repo_id="Viet-Mistral/Vistral-7B-Chat",huggingfacehub_api_token='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')
    # model=HuggingFaceEndpoint(repo_id="NlpHUST/gpt2-vietnamese",huggingfacehub_api_token='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',max_new_tokens=200)
    # model=AnthropicLLM(model="claude-2.1",api_key=os.getenv('ANTHROPIC_API_KEY'),temperature=0)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question",'metadata'])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt,verbose=True)
    return chain

# Hàm xử lý câu hỏi từ người dùng
def user_input(user_question):
    """ ## Hàm xử lý câu hỏi từ người dùng"""

    # Chọn embeddings để tạo vector store
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Phải lựa chọn đúng model đã sử dụng để tạo vector store
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") text-multilingual-embedding-002
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = Embeddingsmodel.HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")
    

    # Load vector store đã lưu với metadata
    #FAISS
    # new_db = Embeddingsmodel.FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) 
    # docs = new_db.similarity_search(user_question,k=2) # Lấy 10  chunk có độ tương đồng về nội dung với câu hỏi của người dùng
    #Pinecone
    new_db= Embeddingsmodel.PineconeVectorStore(index_name=os.getenv('index_name'),embedding=embeddings)

    # Thực hiện filter_by_metadata và similarity_search để lấy các chunk có độ tương đồng với câu hỏi của người dùng
    docs1=filter_by_metadata(user_question,new_db)
    docs2=new_db.similarity_search(query=user_question,k=7) # Lấy 10  chunk có độ tương đồng về nội dung với câu hỏi của người dùng
    # Tài liệu được tổng hợp
    docs=docs1 + docs2
    metadata_combined = "\n".join([str(doc.metadata) for doc in docs]) # Kết hợp metadata của các chunk
    chain = get_conversational_chain() # Khởi tạo chain cho việc hỏi đáp
    # Gửi câu hỏi từ người dùng và các chunk có độ tương đồng cho chain để nhận câu trả lời
    response = chain({"input_documents": docs,"metadata": metadata_combined, "question": user_question}, return_only_outputs=True)
    
    # In câu trả lời
    print('Response là: ', response)
    st.write("Dưới đây là câu trả lời của UITBot: ")
    generate_text_slowly(response['output_text'])
   
    # Để check các doc được truy xuất từ filter_by_metadata và similarity_search
    st.write(" # Documents with metadata")
    st.write(f'## Số lượng RAG đã truy xuất là {len(docs1 + docs2)} docs: ')
    st.write(f'## List {len(docs1)} docs1 nhờ sử dụng filter_by_metadata: ', docs1)
    st.write(f'## List {len(docs2)} docs2 nhờ sử dụng similarity_search: ', docs2)

