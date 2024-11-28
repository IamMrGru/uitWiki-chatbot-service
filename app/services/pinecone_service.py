from langchain_core.documents import Document
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
        self.index_name = 'nestjs'
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=self.api_key,
            index_name=self.index_name,
            embedding=self.embeddings,
        )

    async def upsert_chunk(self, chunk_with_metadata):
        # Prepare the payload for Pinecone
        vectors = [
            Document(
                id=chunk_with_metadata['id'],
                values=chunk_with_metadata['values'],
                metadata=chunk_with_metadata['metadata'],
                page_content=chunk_with_metadata['text'],
            )
        ]

        # Use the Pinecone upsert method
        await self.vectorstore.aadd_documents(vectors)

    async def upsert_chunks(self, chunks_with_metadata):
        await self.vectorstore.aadd_documents(chunks_with_metadata)
