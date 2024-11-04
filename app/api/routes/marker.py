from fastapi import APIRouter, HTTPException, Depends
from app.core.config import settings
from pydantic import BaseModel

router = APIRouter()


class MarkerRequest(BaseModel):
    file_path: str


fpath = '/Users/lap15737-local/Documents/Dev/kltn/uitWiki-chatbot-service/app/static/pdf/ctđt_tmđt_2021.pdf'
fpath1 = '/Users/lap15737-local/Documents/Dev/kltn/uitWiki-chatbot-service/app/static/pdf/QuydinhKLTN_metadata.pdf'


@router.post("/mark-it", response_model=dict)
async def mark_it(body: MarkerRequest):
    try:
        # model = load_all_models()
        # full_text, images, out_meta = convert_single_pdf(fpath1, model)

        return {
            "response": 'ok'
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
