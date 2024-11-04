from fastapi import APIRouter, HTTPException
from app.core.markdown_chunking import markdown_chunking
from app.services.pinecone_service import PineconeService

router = APIRouter()
pinecone_service = PineconeService()


@router.get("/upsert", response_model=dict)
async def upsert():
    try:
        chunks = markdown_chunking(
            "app/static/output/ctđt_tmđt_2021/ctđt_tmđt_2021.md")

        await pinecone_service.upsert_chunks(chunks)

        return {
            "response": "Upserted successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
