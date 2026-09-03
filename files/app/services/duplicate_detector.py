"""
D Desk AI — Duplicate Detection Service
────────────────────────────────────────────
Uses cosine similarity on TF-IDF vectors to detect
duplicate tickets before they are created.
"""

import json
from typing import Optional, Tuple
from app.services.classifier import classifier
from app.services.firestore_db import list_tickets


def find_duplicate(description: str, threshold: float = 0.75) -> Tuple[bool, Optional[str]]:
    """
    Check if a new ticket description is a duplicate of any
    existing open/in_progress ticket in Cloud Firestore.

    Returns:
        tuple: (is_duplicate: bool, duplicate_ticket_id: str or None)
    """
    new_vec = classifier.get_tfidf_vector(description)

    # Get active open/in_progress tickets from Firestore
    active_tickets = list_tickets(include_deleted=False)
    for ex in active_tickets:
        if ex.get("status") in ("open", "in_progress") and ex.get("tfidf_vec"):
            try:
                ex_vec = json.loads(ex["tfidf_vec"]) if isinstance(ex["tfidf_vec"], str) else ex["tfidf_vec"]
                sim = classifier.cosine_similarity(new_vec, ex_vec)
                if sim > threshold:
                    return True, ex.get("ticket_id")
            except (json.JSONDecodeError, TypeError):
                continue

    return False, None


def get_tfidf_json(text: str) -> str:
    """Get JSON-serialisable TF-IDF vector for storage."""
    vec = classifier.get_tfidf_vector(text)
    return json.dumps(vec)
