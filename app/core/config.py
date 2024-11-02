from pydantic_settings import SettingsConfigDict, BaseSettings


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


settings = Settings()  # type: ignore
