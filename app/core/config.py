from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    API_V1_STR: str = "/api/v1"
    GOOGLE_API_KEY: str
    PROJECT_NAME: str = "uitWiki Chatbot"
    WIKI_API_KEY: str
    MONGO_URI: str
    MONGO_DB_NAME: str = "uit-wiki"
    REDIS_ENDPOINT: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_NAMESPACE: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str
    VOYAGEAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: str
    LLAMA_CLOUD_API_KEY: str
    COHERE_API_KEY: str


settings = Settings()  # type: ignore
