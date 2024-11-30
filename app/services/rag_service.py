from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pydantic import SecretStr

from app.core import llm_model
from app.core.config import settings
from app.core.huggingface import huggingface_embedding_model, rerank_docs
from app.core.retriever import (bm25_retriever, hybrid_retriever,
                                retrieve_by_metadata,
                                similarity_search_retriever)

api_key = settings.PINECONE_API_KEY
index_name = settings.PINECONE_INDEX_NAME
namespace = settings.PINECONE_NAMESPACE
google_api_key = settings.GOOGLE_API_KEY


class RAGServices:
    def __init__(self, data):
        self.data = data

    def get_rag(self, user_question):

        # embeddings = huggingface_embedding_model()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=SecretStr(google_api_key))

        new_db = PineconeVectorStore(
            index_name=index_name, embedding=embeddings, pinecone_api_key=api_key, namespace=namespace)

        # Retrieve from the Vector Database
        # retriever1 = similarity_search_retriever(new_db,10) # Semantic Retrieve
        # retriever2= retrieve_by_metadata(new_db,10) # Metadata Retrieve
        # retriever3= hybrid_retriever(retriever1,retriever2) # Hybrid Retrieve
        retriever4 = bm25_retriever(new_db, user_question, 20)  # BM25 Retrieve

        # Get docs from retriever
        # docs4 = retriever4.invoke(user_question)

        # Rerank top 10 documents in the retriever
        reranked_docs = rerank_docs(user_question, retriever4, 10)

        # Retrieved Contexts
        docs = reranked_docs
        num_docs = len(docs)

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs,
                          "metadata": metadata_combined,
                          "question": user_question},
                         return_only_outputs=True)

        return response["output_text"], docs, num_docs
