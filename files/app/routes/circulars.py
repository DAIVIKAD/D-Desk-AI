"""
D Desk AI — Circulars Router backed by Cloud Firestore.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.firestore_db import (
    create_circular as db_create_circular,
    list_circulars as db_list_circulars,
    delete_circular as db_delete_circular,
    get_user_by_username,
)

circulars_bp = APIRouter(tags=["Circulars"])
VALID_EXPIRY_DAYS = {7, 30, 90}


class CircularCreate(BaseModel):
    admin_username: str
    subject: str
    body: str
    target: str = "all-tech"
    priority: str = "normal"
    expires_in_days: Optional[int] = None


def _require_admin(username: str) -> str:
    admin = get_user_by_username((username or "").strip().lower())
    if not admin or admin.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges are required.")
    return admin["username"]


@circulars_bp.post("/api/circulars")
def create_circular_route(req: CircularCreate):
    """Create a new circular/notice."""
    admin_username = _require_admin(req.admin_username)
    if not req.subject or not req.body:
        raise HTTPException(status_code=400, detail="Subject and body are required.")

    if req.expires_in_days and req.expires_in_days not in VALID_EXPIRY_DAYS:
        raise HTTPException(status_code=400, detail="Circular expiry must be 7, 30, or 90 days.")

    circ = db_create_circular(
        admin_username=admin_username,
        subject=req.subject,
        body=req.body,
        target=req.target,
        priority=req.priority,
        expires_in_days=req.expires_in_days,
    )
    return circ


@circulars_bp.get("/api/circulars")
def list_circulars_route(target: Optional[str] = None):
    """List circulars, optionally filtered by target audience."""
    return db_list_circulars(target=target)


@circulars_bp.delete("/api/circulars/{circular_id}")
def delete_circular_route(circular_id: str, admin_username: str):
    """Delete a circular manually from the admin view."""
    _require_admin(admin_username)
    db_delete_circular(str(circular_id), admin_username)
    return {"status": "deleted", "id": circular_id}
