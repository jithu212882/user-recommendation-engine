import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "Automated Recommendation System"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./recommendations.db")


settings = Settings()
