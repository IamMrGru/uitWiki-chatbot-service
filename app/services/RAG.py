from app.core import llm_model
# from app.core.pdf_extraction import get_pdf_text_with_metadata
from app.core import embeddings_model
import os

faiss_index_path = os.path.join(
    os.path.dirname(__file__), '../static/faiss_index/')


class RAGServices:
    def __init__(self, data):
        self.data = data

    def get_rag(self, user_question):

        embeddings = embeddings_model.HuggingFaceEmbeddings(
            model_name="dangvantuan/vietnamese-embedding")

        new_db = embeddings_model.FAISS.load_local(
            faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question, k=10)

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs, "metadata": metadata_combined,
                          "question": user_question}, return_only_outputs=True)

        return response["output_text"]
