import hashlib
import json
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, SecretStr
from redis.asyncio import Redis
from scipy.spatial.distance import cosine

from app.core.config import settings
from app.services.rag_service import RAGServices

router = APIRouter()


class QuestionRequest(BaseModel):
    user_question: str


model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=SecretStr(settings.GOOGLE_API_KEY))


redis = Redis(host=settings.REDIS_ENDPOINT,
              port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD)


def normalize_query(query: str) -> str:
    import re
    query = query.lower().strip()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    return query


def get_cache_key(query: str) -> str:
    normalized_query = normalize_query(query)
    return hashlib.md5(normalized_query.encode('utf-8')).hexdigest()


async def get_similar_cached(query: str):
    normalized_query = normalize_query(query)
    embedding = model.embed_query(normalized_query)

    keys = await redis.keys("cache:qa:*")

    if not keys:
        return None

    for key in keys:
        cached_data = await redis.get(key)
        if cached_data is None:
            continue

        cached_json = json.loads(cached_data)
        cached_embedding = np.array(cached_json['embedding'], dtype=np.float32)

        similarity = 1 - cosine(embedding, cached_embedding)

        print(similarity)

        if similarity > 0.9:
            return cached_json['response']

    return None


@router.post("/send_message", response_model=dict)
async def read_root(body: QuestionRequest):
    try:
        user_question = body.user_question

        cached_response = await get_similar_cached(user_question)
        if cached_response:
            return {
                "cached": True,
                "response": cached_response
            }

        rag_services = RAGServices(data=None)

        response, retrieved_contexts, num_contexts = rag_services.get_rag(
            user_question)

        # Store all data in a single key
        embedding = model.embed_query(normalize_query(user_question))
        current_time = datetime.utcnow().isoformat()
        cache_data = {
            'question': user_question,
            'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
            'response': response,
            'created_at': current_time,
            'updated_at': current_time
        }

        cache_key = f"cache:qa:{get_cache_key(user_question)}"
        await redis.set(cache_key, json.dumps(cache_data), ex=43200)

        return {
            "cached": False,
            "response": response,
            # "num_contexts": num_contexts,
            # "retrieved_contexts": retrieved_contexts
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")
