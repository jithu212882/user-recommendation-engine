import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone, timedelta
from database import SessionLocal, UserActivity, Item
from engine import RecommenderService
from main import favicon, serialize_item


def test_favicon_route():
    resp = favicon()
    assert resp.status_code == 204, f"Expected 204, got {resp.status_code}"
    print("[PASS] test_favicon_route passed (HTTP 204)")


def test_similarity_scores_and_clean_filtering():
    service = RecommenderService()

    with SessionLocal() as db:
        test_user = "test_user_filtering"
        # Reset any prior activity
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()

        # 1. Test cold start (Zero-score fallback check: must NOT be 0.0)
        cold_recs = service.get_recommendations(user_id=test_user, top_k=4, db=db)
        assert len(cold_recs) == 4, "Cold start should return top_k items"
        for item, score in cold_recs:
            assert score >= 0.05, f"Cold start items should not have score < 0.05, got {score}"
            assert score > 0.0, "Cold start items should have non-zero baseline score"

        # 2. Add search activity + interaction with item #1
        db.add(
            UserActivity(
                user_id=test_user,
                item_id=None,
                event_type="search",
                dwell_time=0.0,
                search_query="fragrance perfume luxury scent",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.add(
            UserActivity(
                user_id=test_user,
                item_id=1,
                event_type="cart",
                dwell_time=30.0,
                search_query=None,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

        # 3. Fetch recommendations
        recs = service.get_recommendations(user_id=test_user, top_k=4, db=db)
        assert len(recs) == 4, "Should return top_k recommendations"

        rec_item_ids = [r[0].id for r in recs]
        assert 1 not in rec_item_ids, "Item #1 was interacted with and must be strictly excluded!"

        # 4. Verify API serialization and match scores
        serialized = [serialize_item(entry) for entry in recs]
        for s in serialized:
            assert "similarity_score" in s, "Serialized item should contain similarity_score"
            assert "match_percentage" in s, "Serialized item should contain match_percentage"
            assert isinstance(s["match_percentage"], int), "match_percentage should be an integer"
            assert 5 <= s["match_percentage"] <= 100, f"match_percentage should be >= 5%, got {s['match_percentage']}"

        # Cleanup
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

    print("[PASS] test_similarity_scores_and_clean_filtering passed")


def test_catalog_expansion_and_bags():
    service = RecommenderService()
    assert len(service.items) >= 194, f"Catalog should contain at least 194 items, found {len(service.items)}"

    with SessionLocal() as db:
        test_user = "test_bags_intent"
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()

        # User searches for bags using updated DummyJSON query
        db.add(
            UserActivity(
                user_id=test_user,
                item_id=None,
                event_type="search",
                dwell_time=0.0,
                search_query="womens-bags leather backpack handbag purse",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

        recs = service.get_recommendations(user_id=test_user, top_k=4, db=db)
        assert len(recs) == 4, "Should return 4 recommendations"
        # At least top items should be womens-bags
        top_cats = [r[0].category for r in recs]
        assert "womens-bags" in top_cats, f"womens-bags expected in recommendations, got {top_cats}"
        for item, score in recs:
            assert score >= 0.05, f"Recommended score should be >= 0.05, got {score}"

        # Cleanup
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

    print("[PASS] test_catalog_expansion_and_bags passed")


def test_pivot_behavior():
    """Verify 5 to 7 clicks in a new category decisively overtake previous preferences."""
    service = RecommenderService()

    with SessionLocal() as db:
        test_user = "test_user_pivot"
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()

        # 8 views in groceries
        groceries = db.query(Item).filter(Item.category == "groceries").limit(8).all()
        sports = db.query(Item).filter(Item.category == "sports-accessories").limit(10).all()

        now = datetime.now(timezone.utc)
        for i, g in enumerate(groceries):
            db.add(
                UserActivity(
                    user_id=test_user,
                    item_id=g.id,
                    event_type="view",
                    dwell_time=10.0,
                    created_at=now - timedelta(minutes=30 - i),
                )
            )
        db.commit()

        # Prior recommendations should be groceries/food
        recs_before = service.get_recommendations(test_user, top_k=4, db=db)
        assert any(r[0].category == "groceries" for r in recs_before)

        # User performs 6 clicks in sports-accessories
        for click_idx in range(6):
            s_item = sports[click_idx]
            db.add(
                UserActivity(
                    user_id=test_user,
                    item_id=s_item.id,
                    event_type="view",
                    dwell_time=10.0,
                    created_at=now + timedelta(seconds=(click_idx + 1) * 10),
                )
            )
            db.commit()

        # After 6 clicks, sports-accessories must decisively overtake
        recs_after = service.get_recommendations(test_user, top_k=4, db=db)
        after_cats = [r[0].category for r in recs_after]
        sports_count = sum(1 for c in after_cats if c == "sports-accessories")
        assert sports_count >= 3, f"Expected at least 3/4 sports-accessories after 6 clicks, got {after_cats}"

        # Cleanup
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

    print("[PASS] test_pivot_behavior passed")


if __name__ == "__main__":
    test_favicon_route()
    test_similarity_scores_and_clean_filtering()
    test_catalog_expansion_and_bags()
    test_pivot_behavior()
    print("All production engine tests passed successfully!")
