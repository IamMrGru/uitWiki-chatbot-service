from typing import Dict, Literal, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from tqdm import tqdm

from app.core.markdown_chunking import markdown_chunking
from app.core.process_pdf import process_pdf
from app.core.process_pdf_v2 import process_document, upsert_vector
from app.services.pinecone_service import PineconeService

router = APIRouter()
pinecone_service = PineconeService()


class Metadata(BaseModel):
    name: str = Field(..., description="Name of the metadata field")
    value: str = Field(..., description="Value of the metadata field")


class UpsertRequest(BaseModel):
    documentKey: str = Field(
        ...,
        description="The unique identifier or S3 key for the document"
    )
    metadata: Dict[str, str] = Field(
        ...,
        description="Document metadata including title, author, version, etc."
    )
    parseType: Literal["llama", "ocr"] = Field(
        ...,
        description="Document processing method to use:\n"
        "- llama: For standard PDFs with text content\n"
        "- ocr: For image-based PDFs or documents with complex layouts"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "documentKey": "regulations/2023/academic.pdf",
                "metadata": {
                    "title": "Academic Regulations 2023",
                    "author": "Academic Office",
                    "documentUrl": "https://example.com/document.pdf",
                    "documentId": "reg_2023_001",
                    "version": "1.0"
                },
                "parseType": "ocr"
            }
        }
    }


@router.post(
    "/upsert",
    response_model=dict,
    summary="Upload and process a document",
    description="""
    Process and upload a document to the vector database using either LLAMA or OCR parsing.
    
    - For LLAMA parsing: Processes standard PDFs and converts them to markdown
    - For OCR parsing: Uses AI vision capabilities to analyze document images, extract text content, and generate comprehensive summaries of the visual information. This method is particularly effective for documents with complex layouts, tables, diagrams, and images.
    The document will be chunked and stored with its metadata for later retrieval.
    """,
    response_description="Processing result with confirmation message or processed document data"
)
async def upsert(body: UpsertRequest):  # Removed Field() decorator
    """
    Process and store a document in the vector database.

    Parameters:
    - **documentKey**: S3 key or identifier for the document
    - **metadata**: Document metadata containing:
        - title: Document title
        - author: Document author/department
        - documentUrl: URL for OCR processing (required for OCR)
        - documentId: Unique identifier
        - version: Document version
    - **parseType**: Processing method ('llama' or 'ocr')

    Returns:
    - For LLAMA: Confirmation of successful processing
    - For OCR: Processed document data with page information

    Raises:
    - 400: Invalid request parameters
    - 500: Processing or storage error
    """
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
