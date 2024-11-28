from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

from app.core.config import settings
from app.core.huggingface import huggingface_embedding_model


class PineconeService:
    def __init__(self):
        # self.embeddings = huggingface_embedding_model() # vinai/phobert-base, dangvantuan/vietnamese-embedding,
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004")
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = 'nestjs'
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=self.api_key,
            index_name=self.index_name,
            embedding=self.embeddings,
        )

    async def upsert_chunk(self, chunk_with_metadata):
        await self.vectorstore.aadd_documents([chunk_with_metadata])

    async def upsert_chunks(self, chunks_with_metadata):
        await self.vectorstore.aadd_documents(chunks_with_metadata)
