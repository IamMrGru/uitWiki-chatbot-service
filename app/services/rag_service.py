from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pydantic import SecretStr

from app.core import llm_model
from app.core.config import settings
from app.core.retriever import (bm25_retriever, filter_by_metadata, rerank,
                                similarity_search_retriever)

api_key = settings.PINECONE_API_KEY
index_name = settings.PINECONE_INDEX_NAME
namespace = settings.PINECONE_NAMESPACE
google_api_key = settings.GOOGLE_API_KEY


class RAGServices:
    def __init__(self, data):
        self.data = data

    def get_rag(self, user_question):

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=SecretStr(google_api_key))

        new_db = PineconeVectorStore(
            # namespace là recursive_chunk2, index_name là evaluation
            index_name=index_name, embedding=embeddings, pinecone_api_key=api_key, namespace=namespace)

        retrieve = similarity_search_retriever(new_db, 100)
        docs1 = rerank(retrieve, 10).invoke(user_question)
        # docs1= filter_by_metadata(new_db,user_question,50)
        # docs1 = bm25_retriever(new_db, user_question, 100)

        docs = docs1  # + docs2

        # Trích xuất tất cả page_content từ docs
        all_page_content = [doc.page_content for doc in docs]

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs,
                          "metadata": metadata_combined,
                          "question": user_question},
                         return_only_outputs=True)

        return response["output_text"], all_page_content
