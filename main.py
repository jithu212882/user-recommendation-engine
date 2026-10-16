import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (
    Item,
    SessionLocal,
    UserActivity,
    get_db,
    init_db,
    seed_items_from_public_api,
)
from engine import RecommenderService

# Global recommender service reference
recommender: Optional[RecommenderService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: Seeds database on startup if empty and initializes recommender."""
    global recommender
    init_db()

    with SessionLocal() as db:
        item_count = db.query(Item).count()
        if item_count < 194:
            print(f"Catalog has {item_count} items (< 194). Automatically seeding full catalog from public API...")
            seed_items_from_public_api()
        else:
            print(f"Database contains full catalog ({item_count} items).")

    print("Initializing RecommenderService...")
    recommender = RecommenderService()
    print("RecommenderService ready with TF-IDF catalog.")

    yield
    print("Shutting down recommender application...")


app = FastAPI(
    title="Automated Recommendation System",
    description="Real-time behavioral recommendation engine powered by FastAPI, SQLite, and scikit-learn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class EventCreate(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    item_id: Optional[int] = Field(None, description="Database ID of the interacted item")
    product_id: Optional[int] = Field(None, description="Product ID (alias for item_id)")
    event_type: str = Field(
        ...,
        description="Event type: 'product_view', 'view', 'cart', 'purchase', or 'search'",
    )
    dwell_time: Optional[float] = Field(
        0.0,
        description="Dwell time in seconds spent on the item",
    )
    search_query: Optional[str] = Field(
        None,
        description="Search query string if event_type is 'search'",
    )


def serialize_item(item, score: Optional[float] = None) -> dict:
    """Helper to convert Item ORM instance into clean JSON dict with optional similarity score."""
    if isinstance(item, (tuple, list)):
        item_obj, tuple_score = item[0], item[1]
        score = tuple_score if score is None else score
    else:
        item_obj = item

    sim_score = None
    if score is not None:
        sim_score = round(float(score), 4)
    elif hasattr(item_obj, "similarity_score") and getattr(item_obj, "similarity_score") is not None:
        sim_score = round(float(getattr(item_obj, "similarity_score")), 4)

    data = {
        "id": item_obj.id,
        "external_id": item_obj.external_id,
        "title": item_obj.title,
        "category": item_obj.category,
        "brand": item_obj.brand,
        "description": item_obj.description,
        "price": item_obj.price,
        "image_url": item_obj.image_url,
    }

    if sim_score is not None:
        data["similarity_score"] = sim_score
        data["match_percentage"] = int(round(sim_score * 100))

    return data


# Endpoints
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Returns 204 No Content for favicon requests to prevent 404 errors."""
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to interactive demo dashboard."""
    return RedirectResponse(url="/demo")


@app.post("/api/events", status_code=201, tags=["Events"])
def log_event(event: EventCreate, db: Session = Depends(get_db)):
    """Accepts user interaction events and saves them to the UserActivity table."""
    try:
        target_item_id = event.item_id if event.item_id is not None else event.product_id
        activity = UserActivity(
            user_id=event.user_id.strip(),
            item_id=target_item_id,
            event_type=event.event_type.strip().lower(),
            dwell_time=float(event.dwell_time or 0.0),
            search_query=event.search_query.strip() if event.search_query else None,
            created_at=datetime.now(timezone.utc),
        )
        db.add(activity)
        db.commit()
        db.refresh(activity)

        # Refreshed ranked recommendations under 20ms
        global recommender
        if recommender is None:
            recommender = RecommenderService(db=db)

        recommended_items = recommender.get_recommendations(
            user_id=activity.user_id,
            top_k=4,
            db=db,
        )
        serialized_recommendations = [serialize_item(entry) for entry in recommended_items]

        return {
            "status": "success",
            "message": f"Event '{activity.event_type}' logged successfully",
            "activity_id": activity.id,
            "user_id": activity.user_id,
            "item_id": activity.item_id,
            "product_id": activity.item_id,
            "event_type": activity.event_type,
            "dwell_time": activity.dwell_time,
            "search_query": activity.search_query,
            "recommendations": serialized_recommendations,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log event: {str(e)}")


@app.get("/api/recommendations/{user_id}", tags=["Recommendations"])
def get_user_recommendations(
    user_id: str,
    limit: int = Query(4, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Returns top personalized recommended items for the specified user."""
    global recommender
    if recommender is None:
        recommender = RecommenderService(db=db)

    recommended_items = recommender.get_recommendations(
        user_id=user_id.strip(),
        top_k=limit,
        db=db,
    )
    return [serialize_item(entry) for entry in recommended_items]


@app.get("/api/items", tags=["Catalog"])
def get_catalog_items(
    limit: int = Query(200, ge=1, le=300),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Returns sample catalog items for UI display."""
    query = db.query(Item)
    if category:
        query = query.filter(Item.category.ilike(f"%{category}%"))
    items = query.order_by(Item.id).limit(limit).all()
    return [serialize_item(item) for item in items]


@app.delete("/api/users/{user_id}/reset", tags=["Events"])
def reset_user_activity(user_id: str, db: Session = Depends(get_db)):
    """Helper to clear activity for a specific user to easily re-test cold-start behavior."""
    deleted_count = (
        db.query(UserActivity).filter(UserActivity.user_id == user_id.strip()).delete()
    )
    db.commit()
    return {
        "status": "success",
        "message": f"Cleared {deleted_count} activities for user '{user_id}'",
    }


@app.get("/demo", response_class=HTMLResponse, tags=["Demo UI"])
def demo_ui():
    """Interactive Single-Page Application for testing real-time recommendations."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SmartRecs — Real-time Recommendation Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #09090b;
      --bg-surface: #121215;
      --bg-card: #18181b;
      --bg-card-hover: #202024;
      --bg-muted: #27272a;
      --border-subtle: #27272a;
      --border-muted: #3f3f46;
      --text-main: #f4f4f5;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --accent-blue: #3b82f6;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.5;
    }

    /* Top Navigation */
    header {
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(12px);
      background: rgba(9, 9, 11, 0.92);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.85rem 2rem;
    }

    .nav-container {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
      flex-wrap: wrap;
    }

    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-icon {
      width: 34px;
      height: 34px;
      background: #18181b;
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-main);
    }

    .brand-title {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--text-main);
    }

    .brand-subtitle {
      font-size: 0.7rem;
      color: var(--text-dim);
      font-weight: 500;
      letter-spacing: 0.02em;
    }

    .nav-controls {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .user-select-wrap {
      display: flex;
      align-items: center;
      background: #18181b;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 0.35rem 0.75rem;
      gap: 0.5rem;
      transition: border-color 0.15s ease;
    }

    .user-select-wrap:hover {
      border-color: var(--border-muted);
    }

    .user-label {
      font-size: 0.74rem;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    select#userSelector {
      background: transparent;
      color: var(--text-main);
      border: none;
      outline: none;
      font-family: inherit;
      font-weight: 600;
      font-size: 0.84rem;
      cursor: pointer;
    }

    select#userSelector option {
      background: #18181b;
      color: #f4f4f5;
    }

    .btn-secondary {
      background: #18181b;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 0.45rem 0.85rem;
      border-radius: var(--radius-sm);
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .btn-secondary:hover {
      background: #27272a;
      border-color: var(--border-muted);
      color: var(--text-main);
    }

    /* Main Container */
    main {
      max-width: 1400px;
      margin: 1.5rem auto 3rem;
      padding: 0 2rem;
      display: grid;
      grid-template-columns: 1fr;
      gap: 2rem;
    }

    /* Intent & Search Section */
    .intent-hero {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.5rem 1.75rem;
      position: relative;
    }

    .hero-heading {
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 0.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-main);
    }

    .hero-heading svg {
      color: var(--text-muted);
    }

    .hero-desc {
      font-size: 0.84rem;
      color: var(--text-muted);
      margin-bottom: 1.25rem;
    }

    .search-box-wrap {
      display: flex;
      gap: 0.65rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .search-input-field {
      flex: 1;
      min-width: 280px;
      position: relative;
    }

    .search-input-field input {
      width: 100%;
      background: #18181b;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 0.65rem 0.85rem 0.65rem 2.4rem;
      color: #fff;
      font-size: 0.88rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.15s ease;
    }

    .search-input-field input:focus {
      border-color: var(--border-muted);
      background: #1d1d21;
    }

    .search-icon-svg {
      position: absolute;
      left: 0.85rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      pointer-events: none;
      display: flex;
      align-items: center;
    }

    .btn-search {
      background: #27272a;
      color: #f4f4f5;
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-sm);
      padding: 0.65rem 1.25rem;
      font-weight: 600;
      font-size: 0.84rem;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .btn-search:hover {
      background: #3f3f46;
      color: #fff;
    }

    .btn-search:disabled,
    .chip:disabled,
    .seg-btn:disabled,
    .btn-cart-minimal:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      pointer-events: none;
    }

    /* Chips */
    .chip-container {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      margin-top: 0.9rem;
      flex-wrap: wrap;
    }

    .chip-label {
      font-size: 0.72rem;
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .chip {
      background: #18181b;
      border: 1px solid var(--border-subtle);
      border-radius: 9999px;
      padding: 0.25rem 0.7rem;
      font-size: 0.74rem;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .chip:hover {
      background: #27272a;
      border-color: var(--border-muted);
      color: var(--text-main);
    }

    /* Dashboard Layout */
    .content-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 2.25rem;
    }

    /* Section Headers */
    .section-head {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      margin-bottom: 1.15rem;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.75rem;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .section-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.55rem;
      color: var(--text-main);
    }

    .badge-recs {
      background: #27272a;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .section-subtext {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    /* Catalog Category Tabs */
    .catalog-tabs-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
      flex-wrap: wrap;
    }

    .catalog-tabs-list {
      display: inline-flex;
      align-items: center;
      background: #18181b;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 0.25rem;
      gap: 0.25rem;
      flex-wrap: wrap;
    }

    .catalog-tab {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.35rem 0.8rem;
      border-radius: calc(var(--radius-sm) - 2px);
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: inherit;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }

    .catalog-tab:hover {
      color: #f4f4f5;
      background: #27272a;
    }

    .catalog-tab.active {
      background: #27272a;
      border-color: var(--border-muted);
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    }

    .catalog-count-badge {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: #18181b;
      border: 1px solid var(--border-subtle);
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
    }

    /* Recommendations Grid */
    .recs-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1.25rem;
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      position: relative;
      cursor: pointer;
    }

    .card:hover {
      background: var(--bg-card-hover);
      border-color: var(--border-muted);
      transform: translateY(-2px);
    }

    @keyframes cardPulseGlow {
      0% {
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
        border-color: var(--accent-blue);
      }
      50% {
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.4), 0 8px 24px rgba(59, 130, 246, 0.25);
        border-color: var(--accent-blue);
        transform: translateY(-2px);
      }
      100% {
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
        border-color: var(--border-muted);
        transform: translateY(0);
      }
    }

    .card.card-engaged-active {
      animation: cardPulseGlow 0.9s cubic-bezier(0.4, 0, 0.2, 1) forwards;
      border-color: var(--accent-blue) !important;
    }

    .rec-card {
      border: 1px solid var(--border-muted);
    }

    .card-rank {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(24, 24, 27, 0.9);
      border: 1px solid var(--border-muted);
      color: var(--text-muted);
      width: 24px;
      height: 24px;
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.72rem;
      font-weight: 700;
      z-index: 2;
    }

    /* Subtle Muted Pill Match Badge */
    .card-match-badge {
      position: absolute;
      top: 10px;
      right: 10px;
      background: #27272a;
      border: 1px solid var(--border-muted);
      color: #d4d4d8;
      padding: 0.18rem 0.55rem;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      z-index: 2;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }

    .card-match-badge.high-match {
      background: rgba(16, 185, 129, 0.12);
      border-color: rgba(16, 185, 129, 0.3);
      color: #34d399;
    }

    .card-match-badge.mid-match {
      background: #27272a;
      border-color: var(--border-muted);
      color: #e4e4e7;
    }

    .card-match-badge.cold-match {
      background: #18181b;
      border-color: var(--border-subtle);
      color: #71717a;
    }

    .card-thumb-wrap {
      width: 100%;
      height: 165px;
      background: #121215;
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border-bottom: 1px solid var(--border-subtle);
    }

    .card-thumb-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      padding: 10px;
      transition: transform 0.2s ease;
    }

    .card:hover .card-thumb-wrap img {
      transform: scale(1.03);
    }

    .card-body {
      padding: 1rem;
      display: flex;
      flex-direction: column;
      flex: 1;
      gap: 0.55rem;
    }

    .card-category-brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      font-size: 0.72rem;
    }

    .category-badge {
      background: #27272a;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 0.12rem 0.45rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .brand-name {
      color: var(--text-dim);
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .card-title {
      font-size: 0.92rem;
      font-weight: 700;
      line-height: 1.35;
      color: var(--text-main);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      height: 2.7em;
    }

    .card-desc {
      font-size: 0.78rem;
      color: var(--text-muted);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      height: 2.4em;
    }

    .card-footer {
      margin-top: auto;
      padding-top: 0.65rem;
      border-top: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-price {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
    }

    /* Card Action Button Group */
    .card-btn-group {
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .btn-cart-minimal {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.35rem;
      background: #27272a;
      border: 1px solid var(--border-muted);
      color: #f4f4f5;
      font-size: 0.76rem;
      font-weight: 600;
      padding: 0.35rem 0.65rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: inherit;
    }

    .btn-cart-minimal:hover {
      background: #3f3f46;
      color: #ffffff;
      border-color: #52525b;
    }

    .btn-secondary-action {
      background: #27272a;
      border: 1px solid var(--border-muted);
      color: #f4f4f5;
      font-size: 0.76rem;
      font-weight: 600;
      padding: 0.35rem 0.75rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: inherit;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .btn-secondary-action:hover {
      background: #3f3f46;
      color: #ffffff;
    }

    /* Product Details Modal (Realistic Dwell Time) */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.78);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease;
    }

    .modal-backdrop.modal-open {
      opacity: 1;
      pointer-events: auto;
    }

    .modal-card {
      background: #18181b;
      border: 1px solid var(--border-muted);
      border-radius: var(--radius-lg);
      width: 100%;
      max-width: 680px;
      max-height: 90vh;
      overflow-y: auto;
      position: relative;
      box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.85);
      transform: scale(0.96) translateY(8px);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .modal-backdrop.modal-open .modal-card {
      transform: scale(1) translateY(0);
    }

    .modal-close-btn {
      position: absolute;
      top: 14px;
      right: 14px;
      background: #27272a;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      z-index: 10;
      transition: all 0.15s ease;
    }

    .modal-close-btn:hover {
      background: #3f3f46;
      color: #fff;
    }

    .modal-layout {
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 1.5rem;
      padding: 1.75rem;
    }

    @media (max-width: 640px) {
      .modal-layout {
        grid-template-columns: 1fr;
        padding: 1.25rem;
      }
    }

    .modal-image-pane {
      background: #121215;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      padding: 1rem;
    }

    .modal-image-pane img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    .modal-info-pane {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .modal-meta-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .dwell-live-badge {
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      font-size: 0.7rem;
      font-weight: 600;
      border-radius: 9999px;
      padding: 0.15rem 0.55rem;
      margin-left: auto;
      font-family: ui-monospace, monospace;
    }

    .modal-title {
      font-size: 1.2rem;
      font-weight: 700;
      line-height: 1.3;
      color: var(--text-main);
    }

    .modal-price {
      font-size: 1.35rem;
      font-weight: 800;
      color: #34d399;
    }

    .modal-description {
      font-size: 0.85rem;
      line-height: 1.6;
      color: #d4d4d8;
    }

    .modal-actions-row {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-top: auto;
      padding-top: 1rem;
    }

    .btn-primary-action {
      background: #3b82f6;
      color: #fff;
      border: 1px solid #2563eb;
      border-radius: var(--radius-sm);
      padding: 0.55rem 1.15rem;
      font-weight: 600;
      font-size: 0.82rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      transition: all 0.15s ease;
      font-family: inherit;
    }

    .btn-primary-action:hover {
      background: #2563eb;
    }

    .btn-primary-action.added {
      background: #10b981;
      border-color: #059669;
    }

    /* Activity Feed Panel */
    .activity-feed-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.25rem 1.5rem;
      margin-top: 1.5rem;
    }

    .feed-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.85rem;
    }

    .feed-title {
      font-size: 0.92rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-main);
    }

    .status-indicator {
      font-size: 0.72rem;
      color: var(--accent-emerald);
      font-weight: 500;
    }

    .feed-list {
      max-height: 180px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.76rem;
    }

    .feed-item {
      padding: 0.4rem 0.65rem;
      border-radius: var(--radius-sm);
      background: #18181b;
      border: 1px solid var(--border-subtle);
      border-left: 3px solid var(--border-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .feed-item.feed-search {
      border-left-color: var(--accent-blue);
    }

    .feed-item.feed-engaged {
      border-left-color: var(--accent-emerald);
    }

    .feed-item.feed-cart {
      border-left-color: #a855f7;
    }

    .feed-item.feed-quick {
      border-left-color: var(--accent-amber);
    }

    .feed-item.feed-product_view {
      border-left-color: #38bdf8;
    }

    .feed-item.feed-view {
      border-left-color: #6366f1;
    }

    @keyframes recsPulse {
      0% { opacity: 0.55; transform: scale(0.995); }
      100% { opacity: 1; transform: scale(1); }
    }

    .recs-updated {
      animation: recsPulse 0.28s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Toast Notification */
    #toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #18181b;
      border: 1px solid var(--border-muted);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      color: #f4f4f5;
      padding: 0.7rem 1.15rem;
      border-radius: var(--radius-sm);
      font-size: 0.82rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transform: translateY(140%);
      transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 9999;
    }

    #toast svg {
      color: var(--accent-emerald);
      flex-shrink: 0;
    }

    #toast.show {
      transform: translateY(0);
    }

    @media (max-width: 768px) {
      header {
        padding: 0.75rem 1rem;
      }
      main {
        padding: 0 1rem;
      }
      .nav-controls {
        width: 100%;
        justify-content: space-between;
      }
    }
  </style>
</head>
<body>

  <!-- Top Navigation -->
  <header>
    <div class="nav-container">
      <div class="brand-wrap">
        <div class="brand-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
        </div>
        <div>
          <div class="brand-title">SmartRecs Platform</div>
          <div class="brand-subtitle">Real-Time Personalization System</div>
        </div>
      </div>

      <div class="nav-controls">
        <div class="user-select-wrap">
          <span class="user-label">Active User:</span>
          <select id="userSelector" onchange="onUserChange()">
            <option value="user_alpha">User Alpha (Tech Enthusiast)</option>
            <option value="user_beta">User Beta (Beauty & Fragrance)</option>
            <option value="user_gamma">User Gamma (Lifestyle & Fitness)</option>
            <option value="user_cold_tester">User Cold Start (New Visitor)</option>
          </select>
        </div>

        <button class="btn-secondary" onclick="resetActiveUser()" title="Clear events for active user to test cold start">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/>
          </svg>
          <span>Reset Session</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main>
    <!-- Intent & Search Section -->
    <section class="intent-hero">
      <div class="hero-heading">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
        </svg>
        <span>Real-Time Search Intent</span>
      </div>
      <p class="hero-desc">
        Simulate user interactions to evaluate dynamic catalog reranking.
      </p>

      <div class="search-box-wrap">
        <div class="search-input-field">
          <span class="search-icon-svg">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
            </svg>
          </span>
          <input
            type="text"
            id="searchInput"
            placeholder="Search query (e.g. 'perfume fragrance', 'smartwatch fitness', 'laptop display')..."
            onkeydown="if(event.key==='Enter') submitSearch()"
          />
        </div>
        <button class="btn-search" onclick="submitSearch()">
          <span>Log Search Intent</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
          </svg>
        </button>
      </div>

      <!-- Quick Chips -->
      <div class="chip-container">
        <span class="chip-label">Quick Intent:</span>
        <button class="chip" onclick="fillAndSearch('fragrance perfume luxury scent')">Fragrance & Scent</button>
        <button class="chip" onclick="fillAndSearch('laptop computer gaming keyboard')">Computers & Tech</button>
        <button class="chip" onclick="fillAndSearch('beauty lipstick mascara makeup')">Cosmetics & Beauty</button>
        <button class="chip" onclick="fillAndSearch('smartwatch fitness tracker workout')">Smartwatches & Health</button>
        <button class="chip" onclick="fillAndSearch('womens-bags leather backpack handbag purse')">Bags & Luggage</button>
      </div>
    </section>

    <!-- Content Grids -->
    <div class="content-grid">
      <!-- 1. Recommended For You Section -->
      <section>
        <div class="section-head">
          <div>
            <div class="section-title">
              <span>Recommended For You</span>
              <span class="badge-recs">Dynamic Rerank</span>
            </div>
            <p class="section-subtext" id="recsSubtitle">
              Personalized ranking based on user dwell times, query vectors, and engagement history.
            </p>
          </div>
          <button class="btn-secondary" onclick="loadRecommendations()">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M21 21v-5h-5"/>
            </svg>
            <span>Refresh Recommendations</span>
          </button>
        </div>

        <div id="recommendationsGrid" class="recs-grid">
          <!-- Populated dynamically -->
        </div>
      </section>

      <!-- 2. Catalog Section -->
      <section>
        <div class="section-head">
          <div>
            <div class="section-title">
              <span>Catalog Explorer</span>
              <span class="badge-recs">Interactive Feed</span>
            </div>
            <p class="section-subtext">
              Interact with catalog products to record dwell times and evaluate real-time model response.
            </p>
          </div>
        </div>

        <!-- Category Filter Tabs -->
        <div class="catalog-tabs-container">
          <div class="catalog-tabs-list" role="tablist" aria-label="Catalog Categories">
            <button class="catalog-tab active" data-category="all" onclick="setCatalogCategory('all', this)">
              All
            </button>
            <button class="catalog-tab" data-category="tech" onclick="setCatalogCategory('tech', this)">
              Laptops & Tech
            </button>
            <button class="catalog-tab" data-category="fragrances" onclick="setCatalogCategory('fragrances', this)">
              Fragrances
            </button>
            <button class="catalog-tab" data-category="bags" onclick="setCatalogCategory('bags', this)">
              Bags & Luggage
            </button>
            <button class="catalog-tab" data-category="beauty" onclick="setCatalogCategory('beauty', this)">
              Beauty
            </button>
            <button class="catalog-tab" data-category="furniture" onclick="setCatalogCategory('furniture', this)">
              Furniture
            </button>
          </div>
          <span id="catalogCountBadge" class="catalog-count-badge">Loading items...</span>
        </div>

        <div id="catalogGrid" class="recs-grid">
          <!-- Populated dynamically -->
        </div>
      </section>
    </div>

    <!-- Live Telemetry / Event Log -->
    <section class="activity-feed-panel">
      <div class="feed-header">
        <div class="feed-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          <span>Live Interaction Feed</span>
          <span class="status-indicator">● Online</span>
        </div>
        <button class="btn-secondary" style="font-size:0.72rem; padding:0.25rem 0.55rem;" onclick="clearFeed()">
          Clear Feed
        </button>
      </div>
      <div id="feedList" class="feed-list">
        <div class="feed-item">
          <span>[SYSTEM] Session initialized. Ready for events.</span>
          <span style="color:var(--text-dim)">Ready</span>
        </div>
      </div>
    </section>
  </main>

  <!-- Product Details Modal (Realistic Dwell Time) -->
  <div id="productModalBackdrop" class="modal-backdrop" onclick="onBackdropClick(event)">
    <div class="modal-card" onclick="event.stopPropagation()">
      <button class="modal-close-btn" onclick="closeProductModal()" aria-label="Close modal">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
      <div class="modal-layout">
        <div class="modal-image-pane">
          <img id="modalImg" src="" alt="Product Image" onerror="this.src='https://via.placeholder.com/300?text=No+Image'" />
        </div>
        <div class="modal-info-pane">
          <div class="modal-meta-row">
            <span id="modalCategory" class="category-badge">Category</span>
            <span id="modalBrand" class="brand-name">Brand</span>
            <span id="modalTimerBadge" class="dwell-live-badge">Dwell: 0.0s</span>
          </div>
          <div id="modalTitle" class="modal-title">Product Title</div>
          <div id="modalPrice" class="modal-price">$0.00</div>
          <div id="modalDesc" class="modal-description">Product description...</div>
          <div class="modal-actions-row">
            <button id="modalCartBtn" class="btn-primary-action" onclick="onModalAddToCart()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>
              </svg>
              <span>Add to Cart</span>
            </button>
            <button class="btn-secondary-action" onclick="closeProductModal()" style="padding:0.55rem 0.95rem;">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Toast Notification -->
  <div id="toast">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20 6 9 17l-5-5"/>
    </svg>
    <span id="toastMsg">Event recorded</span>
  </div>

  <script>
    let activeUserId = "user_alpha";
    let activeModalItem = null;
    let modalOpenedAt = null;
    let modalTimerInterval = null;

    function onUserChange() {
      activeUserId = document.getElementById("userSelector").value;
      logFeedItem(`Switched active profile to [${activeUserId}]`, "system");
      showToast(`Active user set to: ${activeUserId}`);
      loadRecommendations();
    }

    function showToast(message) {
      const toast = document.getElementById("toast");
      document.getElementById("toastMsg").innerText = message;
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 2800);
    }

    function logFeedItem(text, type = "view") {
      const feed = document.getElementById("feedList");
      const item = document.createElement("div");
      item.className = `feed-item feed-${type}`;
      const timeStr = new Date().toLocaleTimeString();
      item.innerHTML = `<span>[${timeStr}] ${text}</span><span style="color:var(--text-dim)">+logged</span>`;
      feed.insertBefore(item, feed.firstChild);
    }

    function clearFeed() {
      document.getElementById("feedList").innerHTML = "";
    }

    // API Calls
    async function postEvent(eventType, itemId, dwellTime, searchQuery) {
      try {
        const payload = {
          user_id: activeUserId,
          item_id: itemId || null,
          product_id: itemId || null,
          event_type: eventType,
          dwell_time: dwellTime || 0.0,
          search_query: searchQuery || null
        };

        const res = await fetch("/api/events", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        return data;
      } catch (err) {
        console.error("Failed to post event:", err);
      }
    }

    let isSubmittingIntent = false;

    function setIntentControlsDisabled(disabled) {
      const searchBtn = document.querySelector(".btn-search");
      if (searchBtn) searchBtn.disabled = disabled;

      const chips = document.querySelectorAll(".chip");
      chips.forEach(chip => {
        chip.disabled = disabled;
      });

      const input = document.getElementById("searchInput");
      if (input) input.disabled = disabled;
    }

    async function submitSearch() {
      if (isSubmittingIntent) return;
      const input = document.getElementById("searchInput");
      const query = input.value.trim();
      if (!query) return;

      isSubmittingIntent = true;
      setIntentControlsDisabled(true);

      try {
        showToast(`Searching intent: "${query}"`);
        logFeedItem(`User [${activeUserId}] searched "${query}" (weight: 3.5x)`, "search");

        const data = await postEvent("search", null, 0.0, query);
        input.value = "";
        if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
          renderRecommendations(data.recommendations);
        } else {
          await loadRecommendations();
        }
      } catch (err) {
        console.error("Failed to submit search intent:", err);
      } finally {
        isSubmittingIntent = false;
        setIntentControlsDisabled(false);
      }
    }

    function fillAndSearch(query) {
      if (isSubmittingIntent) return;
      document.getElementById("searchInput").value = query;
      submitSearch();
    }

    // Debounced Re-ranking Controller
    let recsDebounceTimer = null;
    function debouncedLoadRecommendations(delay = 50) {
      if (recsDebounceTimer) clearTimeout(recsDebounceTimer);
      recsDebounceTimer = setTimeout(() => {
        loadRecommendations();
      }, delay);
    }

    // Instant Trigger on "View" Click:
    // Immediately logs 'product_view' to /api/events with 2.0x active interest multiplier
    // and elevates related items to top priority without waiting for modal close.
    let lastViewedItemId = null;
    let lastViewTimestamp = 0;

    async function triggerInstantProductView(item) {
      if (!item || !item.id) return;

      const now = Date.now();
      if (lastViewedItemId === item.id && (now - lastViewTimestamp) < 250) {
        return;
      }
      lastViewedItemId = item.id;
      lastViewTimestamp = now;

      // Instant UI Feedback
      showToast(`Viewing "${item.title}" — elevating related recommendations!`);
      logFeedItem(`[${activeUserId}] Viewed #${item.id} "${item.title.slice(0, 20)}" (Active View Intent: 2.0x)`, "product_view");

      try {
        const payload = {
          user_id: activeUserId,
          item_id: item.id,
          product_id: item.id,
          event_type: "product_view",
          dwell_time: 0.0,
          search_query: null
        };

        const res = await fetch("/api/events", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        // Dynamic immediate re-ranking from sub-20ms backend response
        if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
          renderRecommendations(data.recommendations);
        } else {
          debouncedLoadRecommendations(40);
        }
      } catch (err) {
        console.error("Failed to trigger instant product view event:", err);
        debouncedLoadRecommendations(40);
      }
    }

    // Modal & Dwell Time Measurement Logic
    function openProductModal(item) {
      if (!item) return;

      // 1. Instant Trigger on "View" Click (no waiting for modal close or dwell time)
      triggerInstantProductView(item);

      // 2. Open Product Details Modal
      activeModalItem = item;
      modalOpenedAt = Date.now();

      const imgEl = document.getElementById("modalImg");
      imgEl.src = item.image_url || 'https://via.placeholder.com/300';
      imgEl.alt = item.title || 'Product';

      document.getElementById("modalCategory").textContent = item.category || 'General';
      document.getElementById("modalBrand").textContent = item.brand || 'Catalog';
      document.getElementById("modalTitle").textContent = item.title || '';
      document.getElementById("modalPrice").textContent = `$${Number(item.price || 0).toFixed(2)}`;
      document.getElementById("modalDesc").textContent = item.description || 'No additional description available.';

      const cartBtn = document.getElementById("modalCartBtn");
      cartBtn.classList.remove("added");
      cartBtn.disabled = false;
      cartBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>
        </svg>
        <span>Add to Cart</span>
      `;

      const timerBadge = document.getElementById("modalTimerBadge");
      timerBadge.textContent = "Dwell: 0.0s";
      if (modalTimerInterval) clearInterval(modalTimerInterval);
      modalTimerInterval = setInterval(() => {
        if (modalOpenedAt) {
          const elapsed = ((Date.now() - modalOpenedAt) / 1000).toFixed(1);
          timerBadge.textContent = `Dwell: ${elapsed}s`;
        }
      }, 100);

      const backdrop = document.getElementById("productModalBackdrop");
      backdrop.classList.add("modal-open");
    }

    async function closeProductModal() {
      const backdrop = document.getElementById("productModalBackdrop");
      if (!backdrop || !backdrop.classList.contains("modal-open")) return;

      if (modalTimerInterval) {
        clearInterval(modalTimerInterval);
        modalTimerInterval = null;
      }

      const closedAt = Date.now();
      const elapsedSeconds = modalOpenedAt ? parseFloat(((closedAt - modalOpenedAt) / 1000).toFixed(1)) : 0.0;
      const closedItem = activeModalItem;

      backdrop.classList.remove("modal-open");
      activeModalItem = null;
      modalOpenedAt = null;

      if (!closedItem) return;

      // If user had prolonged dwell time (>= 20s), log Engaged Read (2.5x)
      if (elapsedSeconds >= 20.0) {
        showToast(`Engaged view for ${elapsedSeconds.toFixed(1)}s (Logged Engaged Read: 2.5x)`);
        logFeedItem(`[${activeUserId}] Engaged Read #${closedItem.id} "${closedItem.title.slice(0, 20)}" for ${elapsedSeconds.toFixed(1)}s (2.5x)`, "engaged");
        try {
          const data = await postEvent("view", closedItem.id, elapsedSeconds, null);
          if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
            renderRecommendations(data.recommendations);
          } else {
            await loadRecommendations();
          }
        } catch (err) {
          console.error("Failed to log modal engaged view event:", err);
        }
      } else {
        // Normal modal close: product_view was already logged immediately on view click
        logFeedItem(`[${activeUserId}] Closed #${closedItem.id} "${closedItem.title.slice(0, 20)}" (Dwell: ${elapsedSeconds.toFixed(1)}s)`, "view");
      }
    }

    function onBackdropClick(event) {
      if (event.target.id === "productModalBackdrop") {
        closeProductModal();
      }
    }

    async function onModalAddToCart() {
      if (!activeModalItem) return;
      const item = activeModalItem;
      const cartBtn = document.getElementById("modalCartBtn");

      cartBtn.classList.add("added");
      cartBtn.disabled = true;
      cartBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        <span>Added to Cart!</span>
      `;

      showToast(`Added "${item.title}" to Cart! (3.0x Cart Intent)`);
      logFeedItem(`[${activeUserId}] Added to Cart #${item.id} "${item.title.slice(0, 20)}" (weight: 3.0x)`, "cart");

      try {
        const data = await postEvent("cart", item.id, 20.0, null);
        if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
          renderRecommendations(data.recommendations);
        } else {
          await loadRecommendations();
        }
      } catch (err) {
        console.error("Failed to log cart event:", err);
      }
    }

    async function resetActiveUser() {
      if (!confirm(`Clear all interaction history for ${activeUserId} to test Cold Start?`)) return;
      try {
        await fetch(`/api/users/${activeUserId}/reset`, { method: "DELETE" });
        showToast(`Reset interaction history for ${activeUserId}`);
        logFeedItem(`Cleared activity history for [${activeUserId}] (Cold Start mode active)`, "system");
        await loadRecommendations();
      } catch (err) {
        console.error(err);
      }
    }

    // Render Priority Recommendations Feed
    function renderRecommendations(items) {
      const grid = document.getElementById("recommendationsGrid");
      if (!grid) return;

      grid.innerHTML = "";
      if (!items || items.length === 0) {
        grid.innerHTML = '<div style="color:var(--text-muted); padding:1rem; font-size:0.84rem;">No recommendations found.</div>';
        return;
      }

      items.forEach((item, index) => {
        const card = document.createElement("div");
        card.className = "card rec-card";
        card.dataset.itemId = item.id;
        card.dataset.itemTitle = item.title;
        card._itemData = item;

        let matchPct = 0;
        if (typeof item.match_percentage === "number") {
          matchPct = Math.max(0, Math.min(100, item.match_percentage));
        } else if (typeof item.similarity_score === "number") {
          matchPct = Math.max(0, Math.min(100, Math.round(item.similarity_score * 100)));
        }

        if (matchPct < 5) {
          matchPct = 65;
        }

        const badgeClass = matchPct >= 60 ? "high-match" : "mid-match";
        const badgeText = `Match: ${matchPct}%`;

        card.innerHTML = `
          <div class="card-rank">#${index + 1}</div>
          <div class="card-match-badge ${badgeClass}">${badgeText}</div>
          <div class="card-thumb-wrap">
            <img src="${item.image_url || 'https://via.placeholder.com/300'}" alt="${item.title}" onerror="this.src='https://via.placeholder.com/300?text=No+Image'" />
          </div>
          <div class="card-body">
            <div class="card-category-brand">
              <span class="category-badge">${item.category}</span>
              <span class="brand-name">${item.brand || 'Catalog'}</span>
            </div>
            <div class="card-title" title="${item.title}">${item.title}</div>
            <div class="card-desc">${item.description || ''}</div>
            <div class="card-footer">
              <div class="card-price">$${Number(item.price).toFixed(2)}</div>
              <div class="card-btn-group">
                <button class="btn-secondary-action view-details-btn" title="View Product Details">
                  <span>View Details</span>
                </button>
                <button class="btn-cart-minimal direct-cart-btn" title="Add to Cart">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>
                  </svg>
                  <span>Cart</span>
                </button>
              </div>
            </div>
          </div>
        `;

        card.addEventListener("click", () => {
          openProductModal(item);
        });

        const viewBtn = card.querySelector(".view-details-btn");
        if (viewBtn) {
          viewBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            openProductModal(item);
          });
        }

        const directCartBtn = card.querySelector(".direct-cart-btn");
        if (directCartBtn) {
          directCartBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            showToast(`Added "${item.title}" to Cart!`);
            logFeedItem(`[${activeUserId}] Direct Cart #${item.id} "${item.title.slice(0, 20)}" (weight: 3.0x)`, "cart");
            const data = await postEvent("cart", item.id, 20.0, null);
            if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
              renderRecommendations(data.recommendations);
            } else {
              await loadRecommendations();
            }
          });
        }

        grid.appendChild(card);
      });

      // Subtle dynamic pulse animation on recommendation updates
      grid.classList.remove("recs-updated");
      void grid.offsetWidth;
      grid.classList.add("recs-updated");
    }

    // Load Recommendations
    async function loadRecommendations() {
      const grid = document.getElementById("recommendationsGrid");
      if (!grid.children.length) {
        grid.innerHTML = '<div style="color:var(--text-muted); padding:1.5rem; font-size:0.84rem;">Computing recommendations...</div>';
      }

      try {
        const res = await fetch(`/api/recommendations/${activeUserId}?limit=4`);
        const items = await res.json();
        renderRecommendations(items);
      } catch (err) {
        grid.innerHTML = `<div style="color:#f87171; padding:1rem; font-size:0.84rem;">Error loading recommendations: ${err.message}</div>`;
      }
    }

    // Global Catalog State for Interactive Category Filtering
    let allCatalogItems = [];
    let activeCatalogCategory = "all";

    function setCatalogCategory(catKey, btnElement) {
      activeCatalogCategory = catKey;

      // Update active tab button style
      document.querySelectorAll(".catalog-tab").forEach(tab => tab.classList.remove("active"));
      if (btnElement) {
        btnElement.classList.add("active");
      } else {
        const matchingBtn = document.querySelector(`.catalog-tab[data-category="${catKey}"]`);
        if (matchingBtn) matchingBtn.classList.add("active");
      }

      renderCatalogItems();
    }

    function filterItemsByCategory(items, catKey) {
      if (!items || items.length === 0) return [];
      if (catKey === "all") return items;

      return items.filter(item => {
        const cat = (item.category || "").toLowerCase();
        if (catKey === "tech") {
          return cat.includes("laptop") || cat.includes("smartphon") || cat.includes("tablet") || cat.includes("mobile");
        }
        if (catKey === "fragrances") {
          return cat.includes("fragrance");
        }
        if (catKey === "bags") {
          return cat.includes("bag");
        }
        if (catKey === "beauty") {
          return cat.includes("beauty") || cat.includes("skin-care");
        }
        if (catKey === "furniture") {
          return cat.includes("furniture") || cat.includes("home-decoration");
        }
        return cat.includes(catKey);
      });
    }

    function renderCatalogItems() {
      const grid = document.getElementById("catalogGrid");
      if (!grid) return;

      const filtered = filterItemsByCategory(allCatalogItems, activeCatalogCategory);

      // Update items count badge
      const countBadge = document.getElementById("catalogCountBadge");
      if (countBadge) {
        countBadge.textContent = `${filtered.length} item${filtered.length === 1 ? '' : 's'}`;
      }

      grid.innerHTML = "";
      if (filtered.length === 0) {
        grid.innerHTML = '<div style="color:var(--text-muted); padding:2rem; font-size:0.84rem;">No items found in this category.</div>';
        return;
      }

      filtered.forEach(item => {
        const card = document.createElement("div");
        card.className = "card";
        card.dataset.itemId = item.id;
        card.dataset.itemTitle = item.title;
        card._itemData = item;

        card.innerHTML = `
          <div class="card-thumb-wrap">
            <img src="${item.image_url || 'https://via.placeholder.com/300'}" alt="${item.title}" onerror="this.src='https://via.placeholder.com/300?text=No+Image'" />
          </div>
          <div class="card-body">
            <div class="card-category-brand">
              <span class="category-badge">${item.category}</span>
              <span class="brand-name">${item.brand || 'Catalog'}</span>
            </div>
            <div class="card-title" title="${item.title}">${item.title}</div>
            <div class="card-desc">${item.description || ''}</div>
            <div class="card-footer">
              <div class="card-price">$${Number(item.price).toFixed(2)}</div>
              <div class="card-btn-group">
                <button class="btn-secondary-action view-details-btn" title="View Product Details">
                  <span>View Details</span>
                </button>
                <button class="btn-cart-minimal direct-cart-btn" title="Add to Cart">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>
                  </svg>
                  <span>Cart</span>
                </button>
              </div>
            </div>
          </div>
        `;

        card.addEventListener("click", () => {
          openProductModal(item);
        });

        const viewBtn = card.querySelector(".view-details-btn");
        if (viewBtn) {
          viewBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            openProductModal(item);
          });
        }

        const directCartBtn = card.querySelector(".direct-cart-btn");
        if (directCartBtn) {
          directCartBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            showToast(`Added "${item.title}" to Cart!`);
            logFeedItem(`[${activeUserId}] Direct Cart #${item.id} "${item.title.slice(0, 20)}" (weight: 3.0x)`, "cart");
            const data = await postEvent("cart", item.id, 20.0, null);
            if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
              renderRecommendations(data.recommendations);
            } else {
              await loadRecommendations();
            }
          });
        }

        grid.appendChild(card);
      });
    }

    // Load Catalog Items
    async function loadCatalog() {
      const grid = document.getElementById("catalogGrid");
      grid.innerHTML = '<div style="color:var(--text-muted); padding:1.5rem; font-size:0.84rem;">Loading catalog items...</div>';

      try {
        const res = await fetch("/api/items?limit=200");
        allCatalogItems = await res.json();
        renderCatalogItems();
      } catch (err) {
        grid.innerHTML = `<div style="color:#f87171; padding:1rem; font-size:0.84rem;">Error loading catalog: ${err.message}</div>`;
      }
    }

    // Global event delegation guaranteeing clicks work consistently
    document.addEventListener("click", (e) => {
      if (e.target.closest("#productModalBackdrop") || e.target.closest(".direct-cart-btn")) {
        return;
      }

      const viewBtn = e.target.closest(".view-details-btn");
      if (viewBtn) {
        e.stopPropagation();
        const card = viewBtn.closest(".card");
        if (card && card._itemData) {
          openProductModal(card._itemData);
        }
        return;
      }

      const card = e.target.closest(".card");
      if (card && card._itemData && !e.target.closest("button")) {
        openProductModal(card._itemData);
      }
    });

    // Keyboard shortcuts: ESC closes modal
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeProductModal();
      }
    });

    // Initialization on page load
    window.addEventListener("DOMContentLoaded", () => {
      loadCatalog();
      loadRecommendations();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
