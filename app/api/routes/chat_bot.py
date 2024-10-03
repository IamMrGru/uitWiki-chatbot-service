from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.RAG import RAGServices

router = APIRouter()


class QuestionRequest(BaseModel):
    user_question: str


@router.post("/send_message")
async def read_root(body: QuestionRequest):
    try:
        user_question = body.user_question

        rag_services = RAGServices(data=None)

        response = rag_services.get_rag(user_question)

        return {
            "response": response
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}")
