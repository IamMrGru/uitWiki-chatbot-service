import google.generativeai as genai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings
import os

faiss_index_path = os.path.join(
    os.path.dirname(__file__), '../static/faiss_index/')


genai.configure(api_key='AIzaSyDy0KU0k_EsB-NdGhFqPtbgZ-delrVgsUg')


def get_vector_store_with_metadata(chunks_with_metadata):

    print('embedding file', settings.GOOGLE_API_KEY)

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Có thể lựa chọn model Embedding khác
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-embedding")

    # Chuyển các chunk với metadata thành dạng embedding và lưu vào vector store local
    texts = [chunk["content"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
    # Có thể thay đổi vector store khác
    vector_store = FAISS.from_texts(
        texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_index_path)
