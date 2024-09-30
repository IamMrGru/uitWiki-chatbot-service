from pydantic import BaseModel, Field
from datetime import datetime


def datetime_now_sec():
    return datetime.now().replace(microsecond=0)


class PostModel(BaseModel):
    title: str = Field(...)
    description: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "title": "Post One",
                "description": "This is the description for post one",
            }
        }


class UpdatePostModel(BaseModel):
    title: str = Field(...)
    description: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "title": "Post One",
                "description": "This is the description for post one"
            }
        }


def ResponseModel(data, message):
    return {
        "data": data,
        "code": 200,
        "message": message,
    }


def ErrorResponseModel(error, code, message):
    return {"error": error, "code": code, "message": message}
