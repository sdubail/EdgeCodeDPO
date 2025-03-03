from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_KEY: str = ""


settings = Settings()
