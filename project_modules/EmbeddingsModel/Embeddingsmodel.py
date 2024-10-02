#Embedding Model
import google.generativeai as genai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#Vector Store
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_vector_store_with_metadata(chunks_with_metadata):

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Có thể lựa chọn model Embedding khác
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding")

    # Chuyển các chunk với metadata thành dạng embedding và lưu vào vector store local
    texts = [chunk["content"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas) # Có thể thay đổi vector store khác
    vector_store.save_local('faiss_index')