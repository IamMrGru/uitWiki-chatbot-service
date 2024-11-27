from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

from app.core.config import settings


class PineconeService:
    def __init__(self):
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="dangvantuan/vietnamese-embedding")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004")
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.namespace = settings.PINECONE_NAMESPACE
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=self.api_key,
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace
        )  # namespace là recursive_chunk2, #index_name là evaluation

    async def upsert_chunk(self, chunk_with_metadata):
        await self.vectorstore.aadd_documents([chunk_with_metadata])

    async def upsert_chunks(self, chunks_with_metadata):
        await self.vectorstore.aadd_documents(chunks_with_metadata)
