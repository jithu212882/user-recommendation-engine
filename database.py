import os
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Database configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "recsys.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Item(Base):
    """Product Item Model."""

    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    brand = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False, default=0.0)
    image_url = Column(String(500), nullable=True)

    def __repr__(self):
        return f"<Item(id={self.id}, external_id={self.external_id}, title='{self.title[:20]}')>"


class UserActivity(Base):
    """User Interaction Activity Model."""

    __tablename__ = "user_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), index=True, nullable=False)
    item_id = Column(Integer, index=True, nullable=True)
    event_type = Column(String(50), nullable=False)  # 'view', 'cart', 'purchase', 'search'
    dwell_time = Column(Float, default=0.0, nullable=False)
    search_query = Column(String(255), nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"<UserActivity(id={self.id}, user_id='{self.user_id}', "
            f"event_type='{self.event_type}', item_id={self.item_id})>"
        )


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency helper to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_items_from_public_api(api_url: str = "https://dummyjson.com/products?limit=194") -> int:
    """
    Fetches products from DummyJSON public API and inserts missing items into the items table.

    Returns:
        int: Number of new items inserted.
    """
    init_db()

    print(f"Fetching products from {api_url}...")
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    data = response.json()
    products = data.get("products", [])
    print(f"Fetched {len(products)} products from API.")

    inserted_count = 0
    with SessionLocal() as session:
        # Fetch existing external_ids in one batch for performance
        existing_external_ids = set(
            session.scalars(select(Item.external_id)).all()
        )

        for prod in products:
            ext_id = prod.get("id")
            if ext_id is None or ext_id in existing_external_ids:
                continue

            # Pick image url from thumbnail or first image
            image_url = prod.get("thumbnail") or (
                prod.get("images")[0] if prod.get("images") else ""
            )

            item = Item(
                external_id=ext_id,
                title=prod.get("title", ""),
                category=prod.get("category", ""),
                brand=prod.get("brand"),
                description=prod.get("description", ""),
                price=float(prod.get("price", 0.0)),
                image_url=image_url,
            )
            session.add(item)
            existing_external_ids.add(ext_id)
            inserted_count += 1

        session.commit()

        # Query total count in items table
        total_items = session.query(Item).count()

    print(f"Inserted {inserted_count} new items. Total items in database: {total_items}")
    return inserted_count


if __name__ == "__main__":
    init_db()
    inserted = seed_items_from_public_api()

    # Verification query
    with SessionLocal() as session:
        total_count = session.query(Item).count()
        sample_items = session.query(Item).limit(3).all()

        print("\n--- Database Verification ---")
        print(f"Database File: {DB_FILE} (Exists: {os.path.exists(DB_FILE)})")
        print(f"Total Items Count: {total_count}")
        print("\nSample Items:")
        for it in sample_items:
            print(
                f"- [ID: {it.id}, ExtID: {it.external_id}] {it.title} "
                f"| Category: {it.category} | Brand: {it.brand} | Price: ${it.price}"
            )
