from app.api.routes import chat_bot
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(
    chat_bot.router, prefix="/chat_bot", tags=["Chat bot"])
