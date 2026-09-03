"""
D Desk AI — ML Module
─────────────────────────
Clean public API for the ML sub-package.

Usage:
    from app.ml import predict_ticket, detect_spam, classify_image
"""

from app.ml.predict import predict_ticket
from app.ml.spam import detect_spam
from app.ml.image_model import classify_image

__all__ = ["predict_ticket", "detect_spam", "classify_image"]
