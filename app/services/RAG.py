from app.core import llm_model
import os
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from app.core.config import settings
from pydantic import SecretStr

faiss_index_path = os.path.join(
    os.path.dirname(__file__), '../static/faiss_index/')


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
            index_name=index_name, embedding=embeddings, pinecone_api_key=api_key, namespace=namespace)

        docs1 = new_db.similarity_search(query=user_question, k=100)

        docs2 = llm_model.filter_by_metadata(user_question, new_db)

        docs = docs1 + docs2

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs,
                          "metadata": metadata_combined,
                          "question": user_question},
                         return_only_outputs=True)

        return response["output_text"]
