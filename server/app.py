from fastapi import FastAPI

app = FastAPI()

@app.get("/", tags=["Xin chao"])
async def read_root():
    return {"message": "Welcome to this fantastic app!"}