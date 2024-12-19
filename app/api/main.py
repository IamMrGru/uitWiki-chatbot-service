from fastapi import APIRouter
from app.api.routes import chat_bot, pinecone,query_router


api_router = APIRouter()

api_router.include_router(
    chat_bot.router, prefix="/chat_bot", tags=["Chat bot"])
api_router.include_router(
    pinecone.router, prefix="/pinecone", tags=["Pinecone"]
)
api_router.include_router(
    query_router.router, prefix="/query_router", tags=["Query Router"])
