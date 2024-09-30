from fastapi import FastAPI
from server.routes.post import router as post_router

app = FastAPI()


@app.get("/", tags=["Xin chao"])
async def read_root():
    return {"message": "Welcome to this fantastic app!"}

app.include_router(post_router, tags=["Posts"], prefix="/api/v1")
