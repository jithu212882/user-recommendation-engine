from typing import List, Optional
from pydantic import BaseModel, Field


class Item(BaseModel):
    id: int
    name: str
    category: str
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = ""


class UserPreference(BaseModel):
    user_id: int
    interests: List[str] = Field(default_factory=list)
    interacted_item_ids: List[int] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=5, ge=1, le=50)
    user_interests: List[str] = Field(default_factory=list)
    candidate_items: Optional[List[Item]] = None


class RecommendationResult(BaseModel):
    item_id: int
    name: str
    category: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationResult]
    total_returned: int
