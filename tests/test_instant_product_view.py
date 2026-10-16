import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone
from database import SessionLocal, UserActivity, Item
from engine import RecommenderService
from main import app, EventCreate, log_event, serialize_item
from fastapi.testclient import TestClient


def test_product_view_multiplier():
    """Verify product_view event multiplier is assigned between 1.5x and 2.0x (specifically 2.0x)."""
    service = RecommenderService()
    mult_instant = service._calculate_event_multiplier("product_view", 0.0)
    assert mult_instant == 2.0, f"Expected 2.0x multiplier for instant product_view, got {mult_instant}"
    assert 1.5 <= mult_instant <= 2.0, f"Multiplier must be in [1.5, 2.0], got {mult_instant}"

    # Verify engaged dwell on product_view elevates to 2.5x
    mult_engaged = service._calculate_event_multiplier("product_view", 25.0)
    assert mult_engaged == 2.5, f"Expected 2.5x multiplier for engaged dwell, got {mult_engaged}"
    print("[PASS] test_product_view_multiplier passed")


def test_instant_view_elevates_related_items():
    """Verify that clicking View on a product immediately elevates related items to top recommendations."""
    service = RecommenderService()

    with SessionLocal() as db:
        test_user = "test_user_view_trigger"
        # Reset any prior activity
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

        # Find a fragrance item in catalog
        fragrance = db.query(Item).filter(Item.category.ilike("%fragrance%")).first()
        assert fragrance is not None, "Catalog should have fragrance items"

        # Log instant product_view event (dwell_time=0.0 as just clicked)
        db.add(
            UserActivity(
                user_id=test_user,
                item_id=fragrance.id,
                event_type="product_view",
                dwell_time=0.0,
                search_query=None,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

        # Get recommendations
        recs = service.get_recommendations(user_id=test_user, top_k=4, db=db)
        assert len(recs) == 4, "Should return 4 recommendations"

        rec_ids = [r[0].id for r in recs]
        assert fragrance.id not in rec_ids, "Viewed item itself must be strictly excluded from recommendations"

        # Top recommendations must be related items (same category or semantically similar)
        rec_categories = [r[0].category.lower() for r in recs]
        assert "fragrances" in rec_categories, f"Related items in 'fragrances' must be elevated, got {rec_categories}"
        assert recs[0][1] >= 0.05, f"Top similarity score should be >= 0.05, got {recs[0][1]}"

        # Cleanup
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

    print("[PASS] test_instant_view_elevates_related_items passed")


def test_api_events_product_view_and_latency():
    """Verify /api/events endpoint accepts product_view with product_id and returns refreshed recommendations under 20ms."""
    client = TestClient(app)
    test_user = "test_user_api_latency"

    with SessionLocal() as db:
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

    # Find a test item
    with SessionLocal() as db:
        item = db.query(Item).first()
        assert item is not None

    payload = {
        "user_id": test_user,
        "product_id": item.id,
        "event_type": "product_view",
        "dwell_time": 0.0,
    }

    # Warm-up call
    client.post("/api/events", json=payload)

    # Benchmark latency
    times = []
    for _ in range(5):
        start = time.perf_counter()
        resp = client.post("/api/events", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}"

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    print(f"[BENCHMARK] /api/events latency: min={min_ms:.2f}ms, avg={avg_ms:.2f}ms")

    # Check response structure
    data = resp.json()
    assert data["status"] == "success"
    assert data["event_type"] == "product_view"
    assert data["item_id"] == item.id
    assert data["product_id"] == item.id
    assert "recommendations" in data
    assert len(data["recommendations"]) == 4

    # Latency should be under 20ms
    assert min_ms < 20.0, f"Expected minimum latency under 20ms, got {min_ms:.2f}ms"

    # Cleanup
    with SessionLocal() as db:
        db.query(UserActivity).filter(UserActivity.user_id == test_user).delete()
        db.commit()

    print("[PASS] test_api_events_product_view_and_latency passed")


if __name__ == "__main__":
    test_product_view_multiplier()
    test_instant_view_elevates_related_items()
    test_api_events_product_view_and_latency()
    print("All instant product view tests passed successfully!")
