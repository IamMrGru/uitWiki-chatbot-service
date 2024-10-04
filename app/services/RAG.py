from app.core import llm_model
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

faiss_index_path = os.path.join(
    os.path.dirname(__file__), '../static/faiss_index/')


class RAGServices:
    def __init__(self, data):
        self.data = data

    def get_rag(self, user_question):

        embeddings = HuggingFaceEmbeddings(
            model_name="dangvantuan/vietnamese-embedding")

        new_db = FAISS.load_local(
            faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question, k=10)

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs,
                          "metadata": metadata_combined,
                          "question": user_question},
                         return_only_outputs=True)

        return response["output_text"]
