from fastapi import APIRouter
from app.api.routes import chat_bot, marker, pinecone


api_router = APIRouter()

api_router.include_router(
    chat_bot.router, prefix="/chat_bot", tags=["Chat bot"])
api_router.include_router(
    marker.router, prefix="/marker", tags=["Marker"])
api_router.include_router(
    pinecone.router, prefix="/pinecone", tags=["Pinecone"]
)
