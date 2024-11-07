from fastapi import APIRouter, HTTPException
from app.core.markdown_chunking import markdown_chunking
from app.core.process_pdf import process_pdf
from app.services.pinecone_service import PineconeService
from pydantic import BaseModel
from copy import deepcopy
from langchain_core.documents import Document

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

        chunks = markdown_chunking(md_key, metadata)

        updated_chunks = []
        for chunk in chunks:
            # Tạo một bản sao mới của metadata cho mỗi document
            new_metadata = deepcopy(chunk.metadata)
            # Gán page_content vào metadata['text']
            new_metadata['text'] = chunk.page_content
            # Tạo một document mới với metadata đã cập nhật
            updated_doc = Document(
                page_content=chunk.page_content, metadata=new_metadata)
            updated_chunks.append(updated_doc)

        await pinecone_service.upsert_chunks(updated_chunks)

        return {
            "response": 'Upsert successfully',
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
