from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import (ContextualCompressionRetriever,
                                  EnsembleRetriever)
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAI
from pydantic import SecretStr

from app.core.config import settings

api_key = SecretStr(settings.GOOGLE_API_KEY)
api_key2 = SecretStr(settings.COHERE_API_KEY)


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
            description="Ngày công bố tài liệu, format YYYY hoặc YYYY-MM-DD",
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
        enable_limit=True
    )
    return retriever


def bm25_retriever(new_db, user_input, k_nums=50, k_nums_for_bm25=10):
    """ # Đây là hàm tạo retriever dựa trên mô hình DenseVector+Filter Metadata rồi lọc với mô hình BM25"""
    docs_list1 = similarity_search_retriever(new_db, k_nums).invoke(user_input)
    docs_list2 = retrieve_by_metadata(new_db, k_nums).invoke(user_input)
    # Cách để lọc ra các tài liệu không trùng nhau giữa 2 retriever
    docs = docs_list1 + docs_list2
    combined_docs = []
    seen_ids = set()
    for doc in docs:
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            combined_docs.append(doc)
    retriever = BM25Retriever.from_documents(combined_docs, k=k_nums_for_bm25)
    return retriever


def hybrid_retriever(retriever1, retriever2):
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def rerank_docs(user_question, retriever, top_k=10):  # API key is not working
    """Hàm này sẽ sử dụng mô hình Cohere để sắp xếp lại các tài liệu đã được truy xuất"""
    compressor = CohereRerank(cohere_api_key=api_key2,
                              model="rerank-multilingual-v3.0", top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever)
    reranked_docs = compression_retriever.invoke(user_question)
    return reranked_docs
