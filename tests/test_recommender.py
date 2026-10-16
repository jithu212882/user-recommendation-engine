import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.recommender import recommender_service
from app.models.schemas import Item


def test_recommender_basic():
    recommendations = recommender_service.recommend(
        interests=["headphones", "audio", "music"],
        top_k=3,
    )
    assert len(recommendations) > 0, "Recommendations should not be empty"
    # The top item should be headphones
    top_item = recommendations[0]
    print(f"Top recommendation: {top_item.name} with score {top_item.score}")
    assert top_item.item_id == 1, f"Expected item 1 (Headphones), got {top_item.name}"
    print("Test passed successfully!")


if __name__ == "__main__":
    test_recommender_basic()
