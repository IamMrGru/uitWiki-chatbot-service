import gc
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.api import main as api
from app.core.config import settings
from app.core.database import MongoDBConnection

mongodb = MongoDBConnection()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await mongodb.connect()
        yield
    finally:
        await mongodb.close()
        gc.collect()


async def get_db():
    return mongodb.db


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

handler = Mangum(app)


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello World"}


app.include_router(api.api_router, prefix=settings.API_V1_STR)
