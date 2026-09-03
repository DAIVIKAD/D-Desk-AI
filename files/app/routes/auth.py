"""
D Desk AI — Auth Router
──────────────────────────────
Handles user authentication (login) backed by Cloud Firestore.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.firestore_db import authenticate_user, create_user

auth_bp = APIRouter(tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str
    password: str


class SecretCreateUserRequest(BaseModel):
    username: str
    password: str
    role: str
    name: str
    dept: Optional[str] = None
    specialization: Optional[str] = None


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

    return auth_result


@auth_bp.post("/api/secret-create-user")
def secret_create_user(req: SecretCreateUserRequest):
    """
    Secret backdoor endpoint triggered by the 4-tap logo gesture on the frontend.
    Allows provisioning of Admin, Employee, or Technician accounts directly.
    """
    try:
        user = create_user(
            admin_username="secret_override",
            username=req.username,
            password=req.password,
            role=req.role,
            name=req.name,
            dept=req.dept,
            specialization=req.specialization,
        )
        return {"status": "created", "user": user}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
