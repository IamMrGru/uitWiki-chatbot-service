from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    API_V1_STR: str = "/api/v1"
    GOOGLE_API_KEY: str
    PROJECT_NAME: str = "uitWiki Chatbot"


settings = Settings()  # type: ignore
