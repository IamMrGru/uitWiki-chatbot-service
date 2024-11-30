from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import (ContextualCompressionRetriever,
                                  EnsembleRetriever)
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAI
from langchain_voyageai import VoyageAIRerank
from pydantic import SecretStr

from app.core.config import settings

api_key = SecretStr(settings.GOOGLE_API_KEY)
api_key2 = SecretStr(settings.VOYAGEAI_API_KEY)


def similarity_search_retriever(new_db, k_number=10):
    """ # Đây là hàm tạo retriever dựa trên mô hình tìm kiếm tương tự bằng Embeddings Cosine Similarity"""
    retriever = new_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_number}
    )
    return retriever


def retrieve_by_metadata(new_db, k_number=10):
    """ # Đây là hàm tạo retriever dựa trên metadata của tài liệu"""
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="Tên của tài liệu chứa thông tin cần truy xuất",
            type="string",
        ),
        AttributeInfo(
            name="author",
            description="Phòng ban quản lý tài liệu",
            type="string",
        ),
        AttributeInfo(
            name="publicdate",
            description="Ngày công bố tài liệu",
            type="string",
        ),
        AttributeInfo(
            name="version",
            description="Phiên bản của tài liệu (Quyết định số...)",
            type="string",
        ),
        AttributeInfo(
            name="description",
            description="Sơ lược nội dung của tài liệu",
            type="string",
        ),
        AttributeInfo(
            name="category",
            description="Loại tài liệu: Quy định, Quy chế, quy trình, hướng dẫn,...",
            type="string",
        ),
        AttributeInfo(
            name="tags",
            description="Các từ khóa liên quan đến đoạn văn cần truy xuất",
            type="string"
        ),
        AttributeInfo(
            name="target",
            description="Đối tượng cần sử dụng, áp dụng",
            type="string"
        ),
        AttributeInfo(
            name="url",
            description="Đường dẫn đến tài liệu",
            type="string"
        ),
    ]
    document_content_description = "Tóm tắt nội dung của tài liệu này để phục vụ cho việc tư vấn sinh viên "
    llm = GoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0, api_key=api_key)

    retriever = SelfQueryRetriever.from_llm(
        llm,
        new_db,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={'k': k_number},
        search_type='similarity',
    )
    return retriever


def bm25_retriever(new_db, user_input, k_nums=10):
    """ # Đây là hàm tạo retriever dựa trên mô hình BM25"""
    docs_list = similarity_search_retriever(new_db, 350).invoke(user_input)
    retriever = BM25Retriever.from_documents(docs_list, k=k_nums)
    return retriever


def hybrid_retriever(retriever1, retriever2):
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2], weights=[0.5, 0.5]
    )
    return ensemble_retriever


# def rerank(retriever, top_k=5): # API key is not working
#     print(api_key2)
#     compressor = VoyageAIRerank(
#         model="rerank-lite-1", voyageai_api_key='', top_k=top_k
#     )
#     compression_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor, base_retriever=retriever
#     )
#     return compression_retriever
