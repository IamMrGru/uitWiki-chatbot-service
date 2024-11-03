from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from app.core.config import settings

api_key = settings.PINECONE_API_KEY
index_name = settings.PINECONE_INDEX_NAME


def get_vector_store_with_metadata(chunks_with_metadata):

    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-embedding")

    texts = [chunk["content"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]

    vectorstore = PineconeVectorStore.from_texts(
        texts, embedding=embeddings, metadatas=metadatas, index_name=index_name)
