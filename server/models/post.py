from pydantic import BaseModel, Field
from datetime import datetime

def datetime_now_sec():
    return datetime.now().replace(microsecond=0)

class PostSchema(BaseModel):
  title: str = Field(...)
  description: str = Field(...)
  date: datetime = Field(default_factory=datetime_now_sec)

  class Config:
    schema_extra = {
      "example": {
        "title": "Post One",
        "description": "This is the description for post one",
        "date": "2021-04-03"
      }
    }


def ResponseModel(data, message):
    return {
        "data": [data],
        "code": 200,
        "message": message,
    }

def ErrorResponseModel(error, code, message):
    return {"error": error, "code": code, "message": message}