"""
D Desk AI — Similarity Service
──────────────────────────────
Finds similar previously resolved tickets using TF-IDF and Cosine Similarity
to suggest possible solutions to users and technicians.
"""

import json
from typing import Any, Dict, List
from app.services.classifier import classifier
from app.ml.predict import predict_ticket
from app.services.firestore_db import list_tickets


def find_similar_resolved_tickets(text: str, limit: int = 3, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
    """
    Find previously resolved/closed tickets that are similar to the provided text.
    """
    if not text or not text.strip():
        return []

    # Get the tf-idf vector
    new_vec = classifier.get_tfidf_vector(text)

    # Predict the category to use as a strong fallback/filter
    prediction = predict_ticket(text)
    predicted_category = prediction.get("category")

    # Query for resolved or closed tickets from Firestore
    all_tickets = list_tickets(include_deleted=False)
    resolved_tickets = [
        t for t in all_tickets
        if t.get("status") in ("resolved", "closed") and t.get("resolution")
    ]

    matches = []
    for ex in resolved_tickets:
        if ex.get("tfidf_vec"):
            try:
                ex_vec = json.loads(ex["tfidf_vec"]) if isinstance(ex["tfidf_vec"], str) else ex["tfidf_vec"]
                sim = classifier.cosine_similarity(new_vec, ex_vec)

                # Boost similarity if categories match exactly
                is_same_category = (ex.get("category") == predicted_category)
                effective_sim = sim + 0.2 if is_same_category else sim

                # For very short queries (like "hardware", "software"), rely heavily on category
                word_count = len(text.split())
                threshold = 0.1 if word_count < 4 else min_similarity

                if effective_sim >= threshold or (is_same_category and word_count < 4):
                    matches.append({
                        "ticket_id": ex.get("ticket_id"),
                        "description": ex.get("description"),
                        "resolution": ex.get("resolution"),
                        "resolved_at": ex.get("resolved_at"),
                        "similarity": round(sim * 100, 1) if sim > 0 else (50.0 if is_same_category else 0.0),
                        "similarity_score": effective_sim,
                        "assigned_to": ex.get("assigned_to"),
                        "category": ex.get("category"),
                    })
            except (json.JSONDecodeError, TypeError):
                continue

    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches[:limit]
