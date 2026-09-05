"""
D Desk AI — Auth Router
──────────────────────────────
Handles user authentication (login) backed by Cloud Firestore.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.firestore_db import authenticate_user

auth_bp = APIRouter(tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str
    password: str


@auth_bp.post("/api/login")
def login(req: LoginRequest):
    """Authenticate a user with username & password against Cloud Firestore."""
    username = req.username.strip()
    password = req.password.strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    auth_result = authenticate_user(username, password)
    if not auth_result:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if auth_result.get("error") == "account_disabled":
        raise HTTPException(status_code=403, detail=auth_result.get("detail", "Account is disabled."))

    return auth_result
