from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_FOLDER = Path(__file__).parents[1]
PYPKG_ROOT_FOLDER = ROOT_FOLDER / "edgecodedpo"
env_file_name = ".env"


class Settings(BaseSettings):
    OPENAI_KEY: str = ""
    HF_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=(PYPKG_ROOT_FOLDER / env_file_name).as_posix(),
        env_file_encoding="utf-8",
    )


settings = Settings()
