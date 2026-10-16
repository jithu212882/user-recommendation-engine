from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from database import Item, SessionLocal, UserActivity, init_db

# Initialize dense embedding model globally on CPU
model = SentenceTransformer("all-MiniLM-L6-v2")


class RecommenderService:
    """Automated Recommender Service powered by SentenceTransformers (all-MiniLM-L6-v2), Cosine Similarity, and Sliding Recency Weighting."""

    WINDOW_SIZE: int = 12
    DECAY_LAMBDA: float = 0.85
    SEARCH_MULTIPLIER: float = 3.5
    ENGAGED_VIEW_MULTIPLIER: float = 2.5
    CART_MULTIPLIER: float = 3.0
    PURCHASE_MULTIPLIER: float = 3.5
    PRODUCT_VIEW_MULTIPLIER: float = 2.0
    STANDARD_VIEW_MULTIPLIER: float = 1.0
    BOUNCE_VIEW_MULTIPLIER: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.05
    CATEGORY_FALLBACK_SCORE: float = 0.72
    COLD_START_BASELINE_SCORE: float = 0.70

    def __init__(self, db: Optional[Session] = None):
        """Load catalog from DB and encode items into dense semantic embeddings."""
        init_db()
        self.items: List[Item] = []
        self.item_id_to_idx: Dict[int, int] = {}
        self.ext_id_to_idx: Dict[int, int] = {}
        self.catalog_embeddings: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None  # Backwards compatibility alias

        self._load_and_fit_catalog(db)

    def _load_and_fit_catalog(self, db: Optional[Session] = None):
        """Loads all catalog items from the database and encodes them with all-MiniLM-L6-v2."""
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # Query all items ordered by id
            self.items = db.query(Item).order_by(Item.id).all()
            if not self.items:
                self.catalog_embeddings = None
                self.tfidf_matrix = None
                return

            self.item_id_to_idx = {item.id: idx for idx, item in enumerate(self.items)}
            self.ext_id_to_idx = {
                item.external_id: idx
                for idx, item in enumerate(self.items)
                if item.external_id is not None
            }

            # Product texts composed of category, brand, title, description
            product_texts = [
                f"{item.category or ''} {item.brand or ''} {item.title or ''} {item.description or ''}".strip()
                for item in self.items
            ]

            # Dense normalized semantic embeddings
            self.catalog_embeddings = model.encode(
                product_texts, convert_to_numpy=True, normalize_embeddings=True
            )
            self.tfidf_matrix = self.catalog_embeddings
        finally:
            if close_session:
                db.close()

    def _calculate_event_multiplier(self, event_type: str, dwell_time: float) -> float:
        """
        Calculates interaction event multiplier relative to standard view (1.0x):
        - 'search': 3.5x multiplier
        - 'cart': 3.0x multiplier
        - 'purchase': 3.5x multiplier
        - 'product_view': 2.0x active interest multiplier on instant view click
        - 'view' (dwell >= 20s): 2.5x multiplier ("Engaged Read")
        - 'view' (dwell < 5s): 0.5x multiplier ("Bounce")
        - 'view' (standard): 1.0x multiplier
        """
        evt = (event_type or "").lower().strip()
        if evt == "search":
            return self.SEARCH_MULTIPLIER
        elif evt == "cart":
            return self.CART_MULTIPLIER
        elif evt == "purchase":
            return self.PURCHASE_MULTIPLIER
        elif evt in ("product_view", "click_view", "view_click"):
            if dwell_time >= 20.0:
                return self.ENGAGED_VIEW_MULTIPLIER
            return self.PRODUCT_VIEW_MULTIPLIER
        elif evt == "view":
            if dwell_time >= 20.0:
                return self.ENGAGED_VIEW_MULTIPLIER
            elif dwell_time < 5.0 and dwell_time > 0.0:
                return self.BOUNCE_VIEW_MULTIPLIER
            return self.STANDARD_VIEW_MULTIPLIER
        return self.STANDARD_VIEW_MULTIPLIER

    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 4,
        db: Optional[Session] = None,
    ) -> List[Tuple[Item, float]]:
        """
        Generate recommendations for a user:
        1. Sliding window over latest 10-12 events with decay factor lambda ≈ 0.85 per step.
        2. Apply 3.5x multiplier for search intent, 2.5x for engaged read, 3.0x for add to cart.
        3. Strict exclusion of interacted items.
        4. Zero-Score Fallback: If similarity is < 0.05, fallback to category-level matching
           from the closest active category.
        """
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # Query all activities for this user ordered newest to oldest
            activities: List[UserActivity] = (
                db.query(UserActivity)
                .filter(UserActivity.user_id == user_id)
                .order_by(UserActivity.created_at.desc())
                .all()
            )

            # Cold start: no user activity exists -> return diverse catalog items with baseline score (no 0%)
            if not activities or self.catalog_embeddings is None or len(self.catalog_embeddings) == 0:
                cold_items = self.items[:top_k]
                for item in cold_items:
                    setattr(item, "similarity_score", self.COLD_START_BASELINE_SCORE)
                return [(item, self.COLD_START_BASELINE_SCORE) for item in cold_items]

            embedding_dim = self.catalog_embeddings.shape[1]
            user_vector = np.zeros(embedding_dim, dtype=np.float32)
            interacted_item_ids: Set[int] = set()
            category_weights: Dict[str, float] = {}

            # Gather all interacted items across all history to ensure strict exclusion
            for act in activities:
                if act.item_id is not None:
                    try:
                        act_id = int(act.item_id)
                        interacted_item_ids.add(act_id)
                        idx = self.item_id_to_idx.get(act_id)
                        if idx is None:
                            idx = self.ext_id_to_idx.get(act_id)
                        if idx is not None:
                            matched_item = self.items[idx]
                            if matched_item.id is not None:
                                interacted_item_ids.add(matched_item.id)
                            if matched_item.external_id is not None:
                                interacted_item_ids.add(matched_item.external_id)
                    except (ValueError, TypeError):
                        pass

            # Sliding window over latest 10 to 12 events
            window_activities = activities[:self.WINDOW_SIZE]

            for step, act in enumerate(window_activities):
                # Gradual recency weighting: lambda ≈ 0.85 per step
                recency_weight = self.DECAY_LAMBDA ** step
                event_mult = self._calculate_event_multiplier(act.event_type or "", act.dwell_time or 0.0)
                step_weight = event_mult * recency_weight

                event_type = (act.event_type or "").lower().strip()

                if event_type == "search":
                    search_query = (act.search_query or "").strip()
                    if search_query:
                        query_vector = model.encode(
                            [search_query], convert_to_numpy=True, normalize_embeddings=True
                        )[0]
                        user_vector += step_weight * query_vector

                        # Associate query with matching categories in catalog for category fallback
                        q_lower = search_query.lower()
                        for it in self.items:
                            if it.category and it.category.lower() in q_lower:
                                category_weights[it.category] = (
                                    category_weights.get(it.category, 0.0) + step_weight * 0.5
                                )
                else:
                    if act.item_id is not None:
                        idx = self.item_id_to_idx.get(act.item_id)
                        if idx is None:
                            idx = self.ext_id_to_idx.get(act.item_id)

                        if idx is not None:
                            matched_item = self.items[idx]
                            item_vec = self.catalog_embeddings[idx]
                            user_vector += step_weight * item_vec

                            if matched_item.category:
                                category_weights[matched_item.category] = (
                                    category_weights.get(matched_item.category, 0.0) + step_weight
                                )

            # Determine closest active category from window
            closest_category = max(category_weights, key=category_weights.get) if category_weights else None

            # Normalize the final User Vector
            norm = float(np.linalg.norm(user_vector))
            recommendations: List[Tuple[Item, float]] = []
            seen_ids = set(interacted_item_ids)

            if norm > 0:
                normalized_user_vector = (user_vector / norm).reshape(1, -1)
                similarities = cosine_similarity(normalized_user_vector, self.catalog_embeddings).flatten()
                ranked_indices = np.argsort(similarities)[::-1]

                for idx in ranked_indices:
                    candidate_item = self.items[idx]
                    if candidate_item.id in seen_ids or candidate_item.external_id in seen_ids:
                        continue

                    raw_score = float(similarities[idx])
                    # If score is >= 0.05, keep it as valid vector match
                    if raw_score >= self.SIMILARITY_THRESHOLD:
                        clean_score = round(min(1.0, max(0.0, raw_score)), 4)
                        setattr(candidate_item, "similarity_score", clean_score)
                        recommendations.append((candidate_item, clean_score))
                        seen_ids.add(candidate_item.id)
                        if len(recommendations) >= top_k:
                            break
                    else:
                        # Below 0.05: stop treating as valid dense match, trigger Zero-Score Fallback
                        break

            # Zero-Score Fallback:
            # If recommendations have fewer than top_k items (or score below 0.05),
            # fallback to category-level matching from the closest active category.
            if len(recommendations) < top_k:
                # 1. Primary fallback: items from closest active category
                if closest_category:
                    cat_candidates = [
                        it for it in self.items
                        if it.category == closest_category
                        and it.id not in seen_ids
                        and it.external_id not in seen_ids
                    ]
                    for cand in cat_candidates:
                        cat_score = self.CATEGORY_FALLBACK_SCORE
                        setattr(cand, "similarity_score", cat_score)
                        recommendations.append((cand, cat_score))
                        seen_ids.add(cand.id)
                        if len(recommendations) >= top_k:
                            break

                # 2. Secondary fallback: items from other active categories
                if len(recommendations) < top_k and category_weights:
                    sorted_cats = [
                        c for c in sorted(category_weights, key=category_weights.get, reverse=True)
                        if c != closest_category
                    ]
                    for sc in sorted_cats:
                        sec_candidates = [
                            it for it in self.items
                            if it.category == sc
                            and it.id not in seen_ids
                            and it.external_id not in seen_ids
                        ]
                        for cand in sec_candidates:
                            sec_score = 0.65
                            setattr(cand, "similarity_score", sec_score)
                            recommendations.append((cand, sec_score))
                            seen_ids.add(cand.id)
                            if len(recommendations) >= top_k:
                                break
                        if len(recommendations) >= top_k:
                            break

                # 3. Final fallback: top catalog items with clean baseline score
                if len(recommendations) < top_k:
                    for cand in self.items:
                        if cand.id not in seen_ids and cand.external_id not in seen_ids:
                            base_score = 0.60
                            setattr(cand, "similarity_score", base_score)
                            recommendations.append((cand, base_score))
                            seen_ids.add(cand.id)
                            if len(recommendations) >= top_k:
                                break

            return recommendations
        finally:
            if close_session:
                db.close()


if __name__ == "__main__":
    from datetime import datetime, timezone

    print("Initializing RecommenderService with all-MiniLM-L6-v2 dense embeddings...")
    service = RecommenderService()
    print(f"Catalog items loaded: {len(service.items)}")
    print(f"Catalog Embeddings shape: {service.catalog_embeddings.shape}")

    # 1. Test Cold Start
    cold_recs = service.get_recommendations(user_id="new_user_123", top_k=4)
    print("\n--- Cold Start Recommendations (User with no history) ---")
    for r, score in cold_recs:
        print(f"- [ID: {r.id}] {r.title} (${r.price}) - {r.category} | Score: {score:.4f}")

    # 2. Simulate User with Search Activity ("fragrance perfume beauty")
    with SessionLocal() as db:
        db.query(UserActivity).filter(UserActivity.user_id == "user_demo_perfume").delete()
        db.add(
            UserActivity(
                user_id="user_demo_perfume",
                item_id=None,
                event_type="search",
                dwell_time=0.0,
                search_query="fragrance perfume luxury scent",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

    search_recs = service.get_recommendations(user_id="user_demo_perfume", top_k=4)
    print("\n--- Search-Based Recommendations (Query: 'fragrance perfume luxury scent') ---")
    for r, score in search_recs:
        pct = int(round(score * 100))
        print(f"- [ID: {r.id}] {r.title} (${r.price}) - {r.category} | Match: {pct}% ({score:.4f})")

    # 3. Simulate User with High Engagement Interaction
    with SessionLocal() as db:
        db.query(UserActivity).filter(UserActivity.user_id == "user_demo_shopper").delete()
        db.add(
            UserActivity(
                user_id="user_demo_shopper",
                item_id=1,
                event_type="cart",
                dwell_time=65.0,
                search_query=None,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

    shopper_recs = service.get_recommendations(user_id="user_demo_shopper", top_k=4)
    print("\n--- Interaction-Based Recommendations (Carted Item #1 with 65s dwell time) ---")
    print("Note: Item #1 must be strictly excluded:")
    for r, score in shopper_recs:
        pct = int(round(score * 100))
        print(f"- [ID: {r.id}] {r.title} (${r.price}) - {r.category} | Match: {pct}% ({score:.4f})")
