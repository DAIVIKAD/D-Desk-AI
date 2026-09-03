"""
D Desk AI — Analytics Router backed by Cloud Firestore.
"""

from fastapi import APIRouter
from app.services.firestore_db import get_category_distribution, get_daily_volume

analytics_bp = APIRouter(tags=["Analytics"])


@analytics_bp.get("/api/analytics/category-distribution")
def category_distribution_route():
    """Get ticket count grouped by category from Cloud Firestore."""
    return get_category_distribution()


@analytics_bp.get("/api/analytics/daily-volume")
def daily_volume_route(days: int = 14):
    """Get daily ticket volume for the last N days from Cloud Firestore."""
    return get_daily_volume(days=days)
