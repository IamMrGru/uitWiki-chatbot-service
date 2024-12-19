import hashlib
import struct

import numpy as np
from fastapi import APIRouter, HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, SecretStr
from redis import Redis
from scipy.spatial.distance import cosine

from app.core.config import settings
from app.services.rag_service import RAGServices

router = APIRouter()


class QuestionRequest(BaseModel):
    user_question: str


redis = Redis(host=settings.REDIS_ENDPOINT, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD,
              decode_responses=True)

model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=SecretStr(settings.GOOGLE_API_KEY))


def normalize_query(query: str) -> str:
    import re
    query = query.lower().strip()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    return query


def get_cache_key(query: str) -> str:
    normalized_query = normalize_query(query)
    embedding = model.embed_query(normalized_query)
    embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
    return hashlib.md5(embedding_bytes).hexdigest()


async def get_similar_cached(query: str):
    normalized_query = normalize_query(query)
    embedding = model.embed_query(normalized_query)  # List[float]
    embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

    # Iterate over keys in Redis
    keys = await redis.keys("*")  # No await on the list; it's already evaluated
    for key in keys:
        # Retrieve cached embedding as bytes
        cached_embedding_bytes = await redis.get(key)
        if cached_embedding_bytes:
            # Convert cached bytes back to a NumPy array
            cached_embedding = np.frombuffer(cached_embedding_bytes, dtype=np.float32)
            
            # Convert the query embedding bytes back to a NumPy array
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Compute similarity
            similarity = 1 - cosine(embedding_array, cached_embedding)
            if similarity > 0.9:
                # Return the cached value if similarity threshold is met
                return await redis.get(key)
    
    # Return None if no similar embedding is found
    return None


@router.post("/send_message", response_model=dict)
async def read_root(body: QuestionRequest):
    try:
        user_question = body.user_question
        # cached_key = await get_similar_cached(user_question)

        # if cached_key:
        #     return {
        #         "cached": True,
        #         "response": cached_key
        #     }

        rag_services = RAGServices(data=None)
        response, retrieved_contexts, num_contexts = rag_services.get_rag(
            user_question)

        return {
            "cached": False,
            "response": response,
            "num_contexts": num_contexts,
            "retrieved_contexts": retrieved_contexts
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
