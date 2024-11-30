from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from redis import Redis

from app.core.config import settings
from app.services.rag_service import RAGServices

router = APIRouter()


class QuestionRequest(BaseModel):
    user_question: str


redis = Redis(host=settings.REDIS_ENDPOINT, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD,
              decode_responses=True)


@router.post("/send_message", response_model=dict)
async def read_root(body: QuestionRequest):
    try:
        user_question = body.user_question
        rag_services = RAGServices(data=None)
        response, retrieved_contexts, num_contexts = rag_services.get_rag(
            user_question)
        return {
            "response": response,
            "num_contexts": num_contexts,
            "retrieved_contexts": retrieved_contexts
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
