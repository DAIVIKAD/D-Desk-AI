"""
D Desk AI — Health Router
────────────────────────────────
System health check endpoint with classifier metadata.
"""

from datetime import datetime
from fastapi import APIRouter
from app.config import Config
from app.services.classifier import classifier
from app.services.firebase_service import get_firebase_status

health_bp = APIRouter()

@health_bp.get("/api/health")
def health():
    """Health check — returns system status, classifier metadata, and Firebase status."""
    model_info = classifier.get_model_info()
    firebase_info = get_firebase_status()

    return {
        "status":           "online",
        "service":          "D Desk AI",
        "version":          "4.0.0",
        "architecture":     "fastapi_microservice",
        "groq_configured":  bool(Config.GROQ_API_KEY),
        "firebase":         firebase_info,
        "timestamp":        datetime.utcnow().isoformat(),
        "classifier": {
            "version":              model_info["version"],
            "method":               model_info["method"],
            "is_trained":           model_info["is_trained"],
            "corpus_size":          model_info["corpus_size"],
            "category_features":    model_info["category_features"],
            "priority_features":    model_info["priority_features"],
            "tfidf_cat_vocab":      model_info["tfidf_cat_vocab_size"],
            "tfidf_pri_vocab":      model_info["tfidf_pri_vocab_size"],
            "categories":           model_info["categories"],
            "priorities":           model_info["priorities"],
        },
    }
