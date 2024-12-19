from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.markdown_chunking import markdown_chunking
from app.core.process_pdf import process_pdf
from app.services.pinecone_service import PineconeService
from langchain_pinecone.vectorstores import PineconeVectorStore
from app.core.config import settings
from langchain_cohere import CohereEmbeddings
from pydantic import SecretStr

router = APIRouter()
pinecone_faq = PineconeService()
pinecone_faq.vectorstore= PineconeVectorStore(
            pinecone_api_key=settings.PINECONE_API_KEY,
            index_name='cohere',
            embedding=CohereEmbeddings(model="embed-multilingual-v3.0",cohere_api_key=SecretStr(settings.COHERE_API_KEY)),
            namespace='faq_questions'
        )

class UpsertRequest(BaseModel):
    faq_query: str


@router.post("/create_faq", response_model=dict)
async def upsert(body: UpsertRequest):
    query = body.faq_query
    try:
        pinecone_faq.upsert_faq(query=query)
        return {
            "query": query,
            "response": 'Upsert FAQ successfully',
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
