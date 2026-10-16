from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import Item, RecommendationResult

# Sample default items catalog for the recommendation engine
DEFAULT_CATALOG: List[Item] = [
    Item(id=1, name="Wireless Noise-Canceling Headphones", category="Electronics", tags=["audio", "wireless", "gadget", "music"], description="High-fidelity audio with active noise cancellation"),
    Item(id=2, name="Smart Fitness Watch", category="Wearables", tags=["fitness", "health", "smartwatch", "gadget"], description="Track your workouts, heart rate, and sleep quality"),
    Item(id=3, name="Ergonomic Mechanical Keyboard", category="Electronics", tags=["typing", "coding", "productivity", "gaming"], description="Custom mechanical switches with ergonomic palm rest"),
    Item(id=4, name="Organic Arabica Coffee Beans", category="Food & Beverage", tags=["coffee", "beverage", "organic", "morning"], description="Rich aroma medium roast freshly sourced organic coffee"),
    Item(id=5, name="Minimalist Leather Backpack", category="Fashion", tags=["travel", "leather", "bag", "minimalist"], description="Durable laptop backpack tailored for modern travelers"),
    Item(id=6, name="4K Ultra-HD Curved Monitor", category="Electronics", tags=["display", "productivity", "gaming", "screen"], description="Immersive ultrawide display for multitasking and entertainment"),
    Item(id=7, name="Stainless Steel Thermal Water Bottle", category="Fitness", tags=["hydration", "fitness", "eco", "travel"], description="Double wall vacuum insulated bottle keeping drinks cold 24h"),
    Item(id=8, name="Machine Learning Fundamentals Book", category="Books", tags=["ai", "coding", "data-science", "learning"], description="Comprehensive handbook covering modern machine learning algorithms"),
]


class ContentBasedRecommender:
    """Content-based recommendation engine utilizing TF-IDF and Cosine Similarity."""

    def __init__(self, catalog: List[Item] = None):
        self.catalog = catalog or DEFAULT_CATALOG
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fit_catalog()

    def _fit_catalog(self):
        corpus = [
            f"{item.name} {item.category} {' '.join(item.tags)} {item.description}"
            for item in self.catalog
        ]
        self.item_vectors = self.vectorizer.fit_transform(corpus)

    def recommend(
        self,
        interests: List[str],
        top_k: int = 5,
        custom_items: List[Item] = None,
    ) -> List[RecommendationResult]:
        items = custom_items or self.catalog
        if not items:
            return []

        # If custom items provided, refit on custom items
        if custom_items:
            corpus = [
                f"{item.name} {item.category} {' '.join(item.tags)} {item.description}"
                for item in items
            ]
            vectorizer = TfidfVectorizer(stop_words="english")
            item_vectors = vectorizer.fit_transform(corpus)
            query_vector = vectorizer.transform([" ".join(interests)])
        else:
            query_str = " ".join(interests) if interests else "all"
            query_vector = self.vectorizer.transform([query_str])
            item_vectors = self.item_vectors

        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, item_vectors).flatten()

        # Sort indices by score descending
        ranked_indices = np.argsort(similarities)[::-1]
        top_indices = ranked_indices[:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            item = items[idx]
            results.append(
                RecommendationResult(
                    item_id=item.id,
                    name=item.name,
                    category=item.category,
                    score=round(score, 4),
                )
            )

        return results


recommender_service = ContentBasedRecommender()
