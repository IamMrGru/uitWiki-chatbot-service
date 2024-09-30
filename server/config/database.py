import motor.motor_asyncio
from .env import Settings
from pymongo.server_api import ServerApi

settings = Settings()  # type: ignore

MONGO_DETAILS = settings.mongo_uri

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client["nextjs-blog"]
