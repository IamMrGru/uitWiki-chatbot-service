from langchain_cohere import CohereEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pydantic import SecretStr

from app.core.config import settings
from app.core.huggingface import huggingface_embedding_model

google_api_key = settings.GOOGLE_API_KEY
cohere_api_key = settings.COHERE_API_KEY


class PineconeService:
    def __init__(self):
        # self.embeddings = huggingface_embedding_model() # vinai/phobert-base, dangvantuan/vietnamese-embedding,
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/text-embedding-004",google_api_key=SecretStr(google_api_key))
        self.embeddings = CohereEmbeddings(
            client="cohere",
            async_client=True,
            model="embed-multilingual-v3.0", cohere_api_key=SecretStr(cohere_api_key))
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = 'cohere'
        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=self.api_key,
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace='bebetter'
        )

    async def upsert_chunk(self, chunk_with_metadata):
        await self.vectorstore.aadd_documents([chunk_with_metadata])

    async def upsert_chunks(self, chunks_with_metadata):
        await self.vectorstore.aadd_documents(chunks_with_metadata)

    def upsert_faq(self, query):
        self.vectorstore.add_texts(texts=[query])
