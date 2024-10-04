import google.generativeai as genai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings
import os

faiss_index_path = os.path.join(
    os.path.dirname(__file__), '../static/faiss_index/')


genai.configure(api_key='AIzaSyDy0KU0k_EsB-NdGhFqPtbgZ-delrVgsUg')


def get_vector_store_with_metadata(chunks_with_metadata):

    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-embedding")

    texts = [chunk["content"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]

    vector_store = FAISS.from_texts(
        texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_index_path)
