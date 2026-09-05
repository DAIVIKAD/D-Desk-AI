"""
D Desk AI — Platform Administration Router
─────────────────────────────────────────────
Secure management endpoints exclusively for Platform Administrators.
Enforces role-based authorization: only users with role "platform_admin"
can access these capabilities.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.firestore_db import (
    authenticate_user,
    create_user,
    delete_user,
    get_platform_audit_info,
    get_user_by_username,
    list_password_reset_requests,
    list_users,
    reset_user_password,
    update_user_role,
    update_user_status,
)

platform_admin_bp = APIRouter(prefix="/api/platform-admin", tags=["Platform Administration"])


class PlatformAdminLoginRequest(BaseModel):
    username: str
    password: str


class CreatePlatformUserRequest(BaseModel):
    admin_username: str
    username: str
    password: str
    role: str
    name: str
    dept: Optional[str] = None
    specialization: Optional[str] = None
    status: str = "active"


class UpdateUserStatusRequest(BaseModel):
    admin_username: str
    status: str  # "active" or "disabled"


class UpdateUserRoleRequest(BaseModel):
    admin_username: str
    role: str
    specialization: Optional[str] = None
    dept: Optional[str] = None


class ForceResetPasswordRequest(BaseModel):
    admin_username: str
    new_password: str
    request_id: Optional[str] = None


class DeleteUserRequest(BaseModel):
    admin_username: str


def _require_platform_admin(admin_username: str) -> dict:
    uname = (admin_username or "").strip().lower()
    if not uname:
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = get_user_by_username(uname)
    if not user:
        raise HTTPException(status_code=401, detail="Administrator account not found.")
    if user.get("status") == "disabled" or user.get("is_disabled") is True:
        raise HTTPException(status_code=403, detail="Platform Admin account is disabled.")
    if user.get("role") != "platform_admin":
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Platform Admin privileges are required to access this resource.",
        )
    return user


def _raise_http(exc: Exception):
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@platform_admin_bp.post("/login")
def platform_admin_login(req: PlatformAdminLoginRequest):
    """
    Authenticate against Cloud Firestore and verify that the account
    has the Platform Admin role.
    """
    username = req.username.strip()
    password = req.password.strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    auth_result = authenticate_user(username, password)
    if not auth_result:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if auth_result.get("error") == "account_disabled":
        raise HTTPException(status_code=403, detail=auth_result.get("detail", "Account is disabled."))

    if auth_result.get("role") != "platform_admin":
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Logged-in account is not authorized as a Platform Admin.",
        )

    return {
        **auth_result,
        "authorized": True,
        "portal": "Platform Administration",
    }


@platform_admin_bp.get("/users")
def get_all_users(admin_username: str = Query(...)):
    """List all accounts across all roles with status, department, and specialization."""
    _require_platform_admin(admin_username)
    try:
        return list_users()
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.post("/users")
def create_account(req: CreatePlatformUserRequest):
    """Create a new Admin, Technician, Employee, or Platform Admin account."""
    _require_platform_admin(req.admin_username)
    try:
        user = create_user(
            admin_username=req.admin_username,
            username=req.username,
            password=req.password,
            role=req.role,
            name=req.name,
            dept=req.dept,
            specialization=req.specialization,
            status=req.status,
        )
        return {"status": "created", "user": user}
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.patch("/users/{username}/status")
def set_user_status(username: str, req: UpdateUserStatusRequest):
    """Enable or disable a user account."""
    _require_platform_admin(req.admin_username)
    try:
        result = update_user_status(
            admin_username=req.admin_username,
            username=username,
            status=req.status,
        )
        return result
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.patch("/users/{username}/role")
def set_user_role(username: str, req: UpdateUserRoleRequest):
    """Update user role and authorization permissions."""
    _require_platform_admin(req.admin_username)
    try:
        result = update_user_role(
            admin_username=req.admin_username,
            username=username,
            role=req.role,
            specialization=req.specialization,
            dept=req.dept,
        )
        return result
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.post("/users/{username}/reset-password")
def force_reset_password(username: str, req: ForceResetPasswordRequest):
    """Force reset any user account password using bcrypt."""
    _require_platform_admin(req.admin_username)
    try:
        user = reset_user_password(
            admin_username=req.admin_username,
            username=username,
            new_password=req.new_password,
            request_id=req.request_id,
        )
        return {"status": "password_reset", "user": user}
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.delete("/users/{username}")
def remove_user(username: str, admin_username: str = Query(...)):
    """Permanently delete a user account and associated requests."""
    _require_platform_admin(admin_username)
    try:
        result = delete_user(admin_username=admin_username, username=username)
        return result
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.get("/reset-requests")
def get_reset_requests(admin_username: str = Query(...), status: Optional[str] = None):
    """List pending or resolved password reset recovery requests."""
    _require_platform_admin(admin_username)
    try:
        return list_password_reset_requests(status=status)
    except Exception as exc:
        _raise_http(exc)


@platform_admin_bp.get("/audit")
def get_audit_telemetry(admin_username: str = Query(...)):
    """Retrieve system health, Firestore database status, user telemetry, and audit logs."""
    _require_platform_admin(admin_username)
    try:
        return get_platform_audit_info(admin_username)
    except Exception as exc:
        _raise_http(exc)
