from copy import deepcopy

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel

from app.core.markdown_chunking import markdown_chunking
from app.core.process_pdf import process_pdf
from app.services.pinecone_service import PineconeService

router = APIRouter()
pinecone_service = PineconeService()


class Metadata(BaseModel):
    name: str
    value: str


class UpsertRequest(BaseModel):
    s3_pdf_key: str
    metadata: list[Metadata]


@router.post("/upsert", response_model=dict)
async def upsert(body: UpsertRequest):
    s3_pdf_key = body.s3_pdf_key
    metadata = body.metadata

    try:
        md_key = await process_pdf(s3_pdf_key)

        if not isinstance(md_key, str):
            raise ValueError(
                "Expected a file path as a string for md_path, got a different type.")

        await markdown_chunking(md_key, metadata)

        return {
            "response": 'Upsert successfully',
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
