from app.core import llm_model
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from app.core.config import settings

faiss_index_path = os.path.join(
    os.path.dirname(__file__), '../static/faiss_index/')


api_key = settings.PINECONE_API_KEY
index_name = settings.PINECONE_INDEX_NAME


class RAGServices:
    def __init__(self, data):
        self.data = data

    def get_rag(self, user_question):

        embeddings = HuggingFaceEmbeddings(
            model_name="dangvantuan/vietnamese-embedding")

        # new_db = FAISS.load_local(
        # faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        new_db = PineconeVectorStore(
            index_name=index_name, embedding=embeddings)

        docs1 = new_db.similarity_search(user_question, k=10)

        docs2 = llm_model.filter_by_metadata(user_question, new_db)

        docs = docs1 + docs2

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs,
                          "metadata": metadata_combined,
                          "question": user_question},
                         return_only_outputs=True)

        return response["output_text"]
