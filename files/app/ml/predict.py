"""
D Desk AI — ML Inference (predict.py)
──────────────────────────────────────────
Provides predict_ticket() for use by FastAPI routes.

Strategy:
  1. On first call, attempt to load saved model.pkl + vectorizer.pkl
     from app/ml/ (written by train.py).
  2. If the .pkl files are absent (e.g. first boot before training),
     fall back transparently to the in-memory EnhancedTicketClassifier
     that is always available in app/services/classifier.
  3. The local model output is ALWAYS final — Groq is never consulted
     inside this module.

Usage:
    from app.ml.predict import predict_ticket

    result = predict_ticket("wifi not connecting")
    # {
    #   "category": "Network",
    #   "priority": "High",
    #   "category_confidence": 0.91,
    #   "priority_confidence": 0.78,
    #   "confidence": 0.91,
    #   "source": "saved_model"   # or "in_memory_classifier"
    # }
"""

import os
import pickle
import warnings

from scipy.sparse import csr_matrix, hstack

from app.ml.features import extract_category_features, extract_priority_features, is_facilities_issue
from app.ml.utils import (
    clean_text,
    get_model_path,
    predict_label_and_confidence,
    preprocess_prediction_text,
)

warnings.filterwarnings("ignore")

# ── Module-level cache so models are loaded only once per process ──────────
_cat_vec   = None
_pri_vec   = None
_cat_model = None
_pri_model = None
_cat_enc   = None
_pri_enc   = None
_source    = None   # "saved_model" | "in_memory_classifier"
# NEW CODE START
_model_name = "Logistic Regression"
# NEW CODE END

# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_models():
    """
    Attempt to load saved .pkl models.
    Sets module globals and returns True on success.
    """
    global _cat_vec, _pri_vec, _cat_model, _pri_model, _cat_enc, _pri_enc, _source, _model_name

    model_path = get_model_path("model.pkl")
    vec_path   = get_model_path("vectorizer.pkl")

    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            with open(model_path, "rb") as f:
                models = pickle.load(f)
            with open(vec_path, "rb") as f:
                vecs = pickle.load(f)

            _cat_vec   = vecs["category_vectorizer"]
            _pri_vec   = vecs["priority_vectorizer"]
            _cat_model = models["category_model"]
            _pri_model = models["priority_model"]
            _cat_enc   = models["category_encoder"]
            _pri_enc   = models["priority_encoder"]
            _source    = "saved_model"
            # NEW CODE START
            _model_name = models.get("model_name", "Logistic Regression")
            # NEW CODE END
            return True
        except Exception:
            pass  # fall through to in-memory fallback

    _source = "in_memory_classifier"
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def predict_ticket(text: str) -> dict:
    """
    Predict category and priority for an IT support ticket text.

    Args:
        text: Raw ticket description.

    Returns:
        dict with keys:
            category, priority,
            category_confidence, priority_confidence, confidence,
            source ("saved_model" | "in_memory_classifier")
    """
    global _source

    raw_text = clean_text(text)
    if not raw_text:
        return {
            "category": "Other", "priority": "Low",
            "category_confidence": 0.0, "priority_confidence": 0.0,
            "confidence": 0.0, "source": "fallback_empty_input",
        }

    # NEW CODE START
    processed_text = preprocess_prediction_text(raw_text)
    facilities_text = f"{raw_text} {processed_text}".strip()
    # NEW CODE END

    # Lazy-load on first call
    if _source is None:
        _load_models()

    # ── Path A: saved .pkl model ──────────────────────────────────────────
    if _source == "saved_model":
        try:
            X_cat = hstack([
                _cat_vec.transform([processed_text]),
                csr_matrix([extract_category_features(processed_text)]),
            ], format="csr")
            X_pri = hstack([
                _pri_vec.transform([processed_text]),
                csr_matrix([extract_priority_features(processed_text)]),
            ], format="csr")

            # NEW CODE START
            category, cat_conf = predict_label_and_confidence(_cat_model, X_cat, _cat_enc)
            priority, pri_conf = predict_label_and_confidence(_pri_model, X_pri, _pri_enc)
            # NEW CODE END

            if is_facilities_issue(facilities_text):
                category = "Other"
                cat_conf = max(cat_conf, 0.92)

            return {
                "category": category,
                "priority": priority,
                "category_confidence": round(cat_conf, 4),
                "priority_confidence": round(pri_conf, 4),
                "confidence": round(cat_conf, 4),
                "source": "saved_model",
                # NEW CODE START
                "model_name": _model_name,
                # NEW CODE END
            }
        except Exception:
            # If the saved model fails mid-way, fall back
            _source = "in_memory_classifier"

    # ── Path B: fall back to in-memory EnhancedTicketClassifier ──────────
    try:
        from app.services.classifier import classifier
        result = classifier.classify(processed_text)
        if is_facilities_issue(facilities_text):
            result["category"] = "Other"
            result["category_confidence"] = max(float(result.get("category_confidence", 0.0)), 0.92)
            result["confidence"] = max(float(result.get("confidence", 0.0)), 0.75)
        return {
            "category": result["category"],
            "priority": result["priority"],
            "category_confidence": round(result.get("category_confidence", 0.0), 4),
            "priority_confidence": round(result.get("priority_confidence", 0.0), 4),
            "confidence": round(result.get("confidence", 0.0), 4),
            "source": "in_memory_classifier",
        }
    except Exception as exc:
        return {
            "category": "Other", "priority": "Low",
            "category_confidence": 0.0, "priority_confidence": 0.0,
            "confidence": 0.0, "source": f"error: {exc}",
        }
