from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    Item,
)
from app.services.recommender import recommender_service, DEFAULT_CATALOG

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Automated Recommendation System"}


@router.get("/catalog", response_model=List[Item], tags=["Catalog"])
def get_catalog():
    """Retrieve the current catalog of candidate items."""
    return DEFAULT_CATALOG


@router.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(payload: RecommendationRequest):
    """
    Generate personalized recommendations based on user interests using TF-IDF and Cosine Similarity.
    """
    try:
        recommendations = recommender_service.recommend(
            interests=payload.user_interests,
            top_k=payload.top_k,
            custom_items=payload.candidate_items,
        )
        return RecommendationResponse(
            user_id=payload.user_id,
            recommendations=recommendations,
            total_returned=len(recommendations),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
