import hashlib
import json
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, SecretStr
from redis.asyncio import Redis

from app.core.config import settings

router = APIRouter()

redis = Redis(
    host=settings.REDIS_ENDPOINT,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD
)


model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=SecretStr(settings.GOOGLE_API_KEY))


class RedisItem(BaseModel):
    question: str
    response: str
    expiration: Optional[int] = 43200  # default 12 hours


@router.post("/set", response_model=dict)
async def set_value(item: RedisItem, no_embedding: bool = False):
    try:
        import re
        normalized_query = item.question.lower().strip()
        normalized_query = re.sub(r"[^a-z0-9\s]", "", normalized_query)
        embedding = model.embed_query(normalized_query)

        current_time = datetime.utcnow().isoformat()
        cache_data = {
            'question': item.question,
            'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
            'response': item.response,
            'created_at': current_time,
            'updated_at': current_time
        }

        cache_key = f"cache:qa:{hashlib.md5(normalized_query.encode('utf-8')).hexdigest()}"
        if item.expiration:
            await redis.set(cache_key, json.dumps(cache_data), ex=item.expiration)
        else:
            await redis.set(cache_key, json.dumps(cache_data))

        response = {
            "message": "Cache entry created successfully",
            "key": cache_key,
            "question": item.question,
            "response": item.response,
            "created_at": current_time,
            "updated_at": current_time
        }
        if not no_embedding:
            response["embedding"] = cache_data["embedding"]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{key}", response_model=dict)
async def get_value(key: str, no_embedding: bool = False):
    try:
        if not key.startswith('cache:qa:'):
            key = f"cache:qa:{key}"

        value = await redis.get(key)
        if value is None:
            raise HTTPException(
                status_code=404, detail="Cache entry not found")

        cached_data = json.loads(value)
        response = {
            "key": key,
            "question": cached_data.get('question'),
            "response": cached_data.get('response'),
            "created_at": cached_data.get('created_at'),
            "updated_at": cached_data.get('updated_at')
        }
        if not no_embedding:
            response["embedding"] = cached_data.get('embedding')
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keys", response_model=dict)
async def get_all_keys(no_embedding: bool = False):
    try:
        keys = await redis.keys("cache:qa:*")
        result = []
        for key in keys:
            cached_data = await redis.get(key)
            if cached_data:
                data = json.loads(cached_data)
                item = {
                    "key": key.decode('utf-8'),
                    "question": data.get('question', 'N/A'),
                    "response": data.get('response', 'N/A'),
                    "created_at": data.get('created_at'),
                    "updated_at": data.get('updated_at')
                }
                if not no_embedding:
                    item["embedding"] = data.get('embedding')
                result.append(item)
        return {"cached_items": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update/{key}", response_model=dict)
async def update_value(key: str, item: RedisItem, no_embedding: bool = False):
    try:
        if not key.startswith('cache:qa:'):
            key = f"cache:qa:{key}"

        exists = await redis.exists(key)
        if not exists:
            raise HTTPException(
                status_code=404, detail="Cache entry not found")

        # Get existing data to preserve created_at
        existing_data = json.loads(await redis.get(key))

        import re
        normalized_query = item.question.lower().strip()
        normalized_query = re.sub(r"[^a-z0-9\s]", "", normalized_query)
        embedding = model.embed_query(normalized_query)

        cache_data = {
            'question': item.question,
            'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
            'response': item.response,
            'created_at': existing_data.get('created_at'),
            'updated_at': datetime.utcnow().isoformat()
        }
        if item.expiration:
            await redis.set(cache_key, json.dumps(cache_data), ex=item.expiration)
        else:
            await redis.set(cache_key, json.dumps(cache_data))

        response = {
            "message": "Cache entry updated successfully",
            "key": key,
            "question": item.question,
            "response": item.response,
            "created_at": cache_data['created_at'],
            "updated_at": cache_data['updated_at']
        }
        if not no_embedding:
            response["embedding"] = cache_data["embedding"]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
