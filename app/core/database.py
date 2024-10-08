import motor.motor_asyncio
from app.core.config import settings


class MongoDBConnection:
    def __init__(self):
        # Asynchronous MongoDB client
        self.client = None
        self.db = None

    async def connect(self):
        """Connect to the MongoDB server asynchronously"""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGO_URI
        )
        self.db = self.client[settings.MONGO_DB_NAME]
        print("INFO:     MongoDB connection established")

    async def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            print("INFO:     MongoDB connection closed")
