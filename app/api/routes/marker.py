from fastapi import APIRouter, HTTPException, Depends
from app.core.config import settings
from pydantic import BaseModel
from marker.convert import convert_single_pdf
from marker.models import load_all_models

router = APIRouter()


class MarkerRequest(BaseModel):
    file_path: str


fpath = '/Users/daogiahai/Documents/Dev/uitWiki-chatbot/app/static/pdf/ctđt_tmđt_2021.pdf'


@router.post("/mark-it", response_model=dict)
async def mark_it(body: MarkerRequest):
    try:
        model = load_all_models()
        full_text, images, out_meta = convert_single_pdf(fpath, model)

        return {
            "response": {
                "full_text": full_text,
                "images": images,
                "out_meta": out_meta
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
