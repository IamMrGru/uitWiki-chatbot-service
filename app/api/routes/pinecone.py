from fastapi import APIRouter, HTTPException
from app.core.markdown_chunking import markdown_chunking
from app.core.process_pdf import process_pdf
from app.services.pinecone_service import PineconeService
from pydantic import BaseModel

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
        # md_key = await process_pdf(s3_key)

        # if not isinstance(md_key, str):
        #     raise ValueError(
        #         "Expected a file path as a string for md_path, got a different type.")

        chunks = markdown_chunking('markdown/quytrinhsinhvien.md', metadata)

        await pinecone_service.upsert_chunks(chunks)

        return {
            "response": chunks,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
