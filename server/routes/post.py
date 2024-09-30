from server.config.database import database
from fastapi import APIRouter, Body, HTTPException
from server.models.post import PostModel, UpdatePostModel, ResponseModel, ErrorResponseModel
from bson import ObjectId
from datetime import datetime
from pymongo.collection import ReturnDocument

router = APIRouter()


def post_helper(post) -> dict:
    return {
        "id": str(post["_id"]),
        "title": post["title"],
        "description": post["description"],
        "date": post["date"],
    }


post_collection = database.get_collection("Post")


@router.post("/posts", response_model=dict)
async def create_post(post: PostModel):
    post_data = post.model_dump()

    new_post = {
        "title": post_data["title"],
        "description": post_data["description"],
        "date": datetime.now(),
    }

    result = await post_collection.insert_one(new_post)

    created_post = await post_collection.find_one({"_id": result.inserted_id})

    return ResponseModel(
        post_helper(created_post),
        'Post created successfully.')


@router.get("/posts")
async def get_posts():
    posts = []
    async for post in post_collection.find():
        posts.append(post_helper(post))
    return ResponseModel(posts, "Data retrieved successfully")


@router.get("/posts/{post_id}", response_model=dict)
async def get_post(post_id: str):
    post = await post_collection.find_one({"_id": ObjectId(post_id)})
    if post:
        return ResponseModel(
            post_helper(post),
            "Post data retrieved successfully")
    return ErrorResponseModel("An error occurred.", 404, "Post doesn't exist.")


@router.patch("/posts/{post_id}", response_model=dict)
async def update_post(post_id: str, post: UpdatePostModel):
    post_updated = await post_collection.find_one_and_update(
        {"_id": ObjectId(post_id)},
        {"$set": post.model_dump()},
        return_document=ReturnDocument.AFTER
    )

    if post_updated is None:
        raise HTTPException(status_code=404, detail="Post not found")

    return ResponseModel(
        post_helper(post_updated),
        "Post updated successfully"
    )


@router.delete("/posts/{post_id}", response_model=dict)
async def delete_post(post_id: str):
    result = await post_collection.delete_one({"_id": ObjectId(post_id)})
    if result.deleted_count == 1:
        return {"status": "Post deleted successfully"}
    return ErrorResponseModel(
        "An error occurred.",
        404,
        "Post with this id doesn't exist.")
