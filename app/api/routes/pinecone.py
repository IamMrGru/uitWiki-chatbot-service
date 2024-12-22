from typing import Dict, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tqdm import tqdm

from app.core.markdown_chunking import markdown_chunking
from app.core.process_pdf import process_pdf
from app.core.process_pdf_v2 import process_document, upsert_vector
from app.services.pinecone_service import PineconeService

router = APIRouter()
pinecone_service = PineconeService()


class Metadata(BaseModel):
    name: str
    value: str


class UpsertRequest(BaseModel):
    documentKey: str
    metadata: Dict[str, str]
    parseType: str


@router.post("/upsert", response_model=dict)
async def upsert(body: UpsertRequest):
    try:
        metadata = body.metadata
        parseType = body.parseType.lower()

        if parseType == "llama":
            s3_pdf_key = body.documentKey

            print(f"Processing PDF from S3: {s3_pdf_key}")

            md_key = await process_pdf(s3_pdf_key)

            if not isinstance(md_key, str):
                raise ValueError(
                    "Expected a file path as a string for md_key, got a different type.")

            await markdown_chunking(md_key, metadata)

            return {"response": "Upsert from PDF processed successfully"}

        elif parseType == "ocr":
            if 'documentUrl' not in body.metadata:
                raise ValueError("documentUrl is required for parseType='url'")

            documentUrl = body.metadata['documentUrl']
            df = process_document(documentUrl)

            if df is None:
                raise ValueError("Document processing returned None")

            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Uploading to Pinecone'):
                pageNumber = row['PageNumber']
                new_metadata = {
                    **metadata,
                    'id': f"{metadata['documentId']}_{pageNumber}",
                    'pageNumber': pageNumber,
                    'text': row['PageText'].replace('\n', ' ')
                }

                await upsert_vector(new_metadata)

            return {"response": df.to_dict(orient="records")}

        else:
            raise ValueError(f"Unsupported parseType: {parseType}")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}"
        )
