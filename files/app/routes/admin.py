"""
Administrative endpoints for user management, password resets, and heatmaps backed by Cloud Firestore.
"""

from collections import Counter, defaultdict
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.firestore_db import (
    create_user,
    delete_user,
    list_password_reset_requests,
    list_users,
    request_password_reset,
    reset_user_password,
    get_user_by_username,
    list_tickets,
    list_image_prediction_records,
    clear_image_prediction_records,
    get_technician_workload,
)

admin_bp = APIRouter(tags=["Admin"])


class CreateUserRequest(BaseModel):
    admin_username: str
    username: str
    password: str
    role: str
    name: str
    dept: Optional[str] = None
    specialization: Optional[str] = None


class DeleteUserRequest(BaseModel):
    admin_username: str
    username: str


class ResetPasswordRequest(BaseModel):
    admin_username: str
    username: str
    new_password: str
    request_id: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    username: str
    note: Optional[str] = None


class ClearPredictionLogsRequest(BaseModel):
    admin_username: str
    ids: Optional[List[str]] = None


def _require_admin(admin_username: str):
    if (admin_username or "").strip().lower() in ("system", "secret_override"):
        return
    admin = get_user_by_username((admin_username or "").strip().lower())
    if not admin or admin.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges are required for this action.")


def _raise_http(exc: Exception):
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@admin_bp.get("/api/admin/users")
def admin_list_users(admin_username: str):
    try:
        _require_admin(admin_username)
        return list_users()
    except Exception as exc:
        _raise_http(exc)


@admin_bp.post("/api/admin/create-user")
def admin_create_user(req: CreateUserRequest):
    try:
        user = create_user(
            admin_username=req.admin_username,
            username=req.username,
            password=req.password,
            role=req.role,
            name=req.name,
            dept=req.dept,
            specialization=req.specialization,
        )
        return {"status": "created", "user": user}
    except Exception as exc:
        _raise_http(exc)


@admin_bp.post("/api/admin/delete-user")
@admin_bp.delete("/api/admin/delete-user")
def admin_delete_user(req: DeleteUserRequest):
    try:
        result = delete_user(admin_username=req.admin_username, username=req.username)
        return {"status": "deleted", **result}
    except Exception as exc:
        _raise_http(exc)


@admin_bp.get("/api/admin/reset-requests")
def admin_list_reset_requests(admin_username: str, status: Optional[str] = None):
    try:
        _require_admin(admin_username)
        return list_password_reset_requests(status=status)
    except Exception as exc:
        _raise_http(exc)


@admin_bp.post("/api/admin/reset-password")
def admin_reset_password(req: ResetPasswordRequest):
    try:
        user = reset_user_password(
            admin_username=req.admin_username,
            username=req.username,
            new_password=req.new_password,
            request_id=req.request_id,
        )
        return {"status": "password_reset", "user": user}
    except Exception as exc:
        _raise_http(exc)


@admin_bp.post("/api/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    try:
        reset_req = request_password_reset(username=req.username, note=req.note)
        return {"status": "requested", "request": reset_req}
    except Exception as exc:
        _raise_http(exc)


@admin_bp.get("/api/admin/heatmap")
def admin_heatmap(admin_username: str):
    try:
        _require_admin(admin_username)
    except Exception as exc:
        _raise_http(exc)

    category_rows = defaultdict(int)
    issue_counter = Counter()
    department_distribution = defaultdict(int)
    department_category_breakdown = defaultdict(lambda: defaultdict(int))
    location_distribution = defaultdict(int)
    location_category_breakdown = defaultdict(lambda: defaultdict(int))

    tickets = list_tickets(include_deleted=False)
    users = {u["username"]: u for u in list_users()}

    for ticket in tickets:
        category = ticket.get("category") or "Other"
        category_rows[category] += 1

        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", (ticket.get("description") or "").lower())
        for token in tokens:
            if token not in {"issue", "problem", "ticket", "need", "with", "that", "this", "from"}:
                issue_counter[token] += 1

        user = users.get(ticket.get("employee_id") or "")
        department = (user.get("dept") if user and user.get("dept") else "Unknown")
        location = (ticket.get("location") or department or "Unknown")
        department_distribution[department] += 1
        department_category_breakdown[department][category] += 1
        location_distribution[location] += 1
        location_category_breakdown[location][category] += 1

    response = dict(category_rows)
    response["category_distribution"] = dict(category_rows)
    response["most_frequent_issues"] = [
        {"issue": issue, "count": count}
        for issue, count in issue_counter.most_common(10)
    ]
    response["department_distribution"] = dict(department_distribution)
    response["department_category_breakdown"] = {
        department: dict(categories)
        for department, categories in department_category_breakdown.items()
    }
    response["location_distribution"] = dict(location_distribution)
    response["location_category_breakdown"] = {
        location: dict(categories)
        for location, categories in location_category_breakdown.items()
    }
    return response


@admin_bp.get("/api/admin/prediction-logs")
def admin_prediction_logs(admin_username: str, limit: int = 100):
    try:
        _require_admin(admin_username)
    except Exception as exc:
        _raise_http(exc)

    return list_image_prediction_records(limit=limit)


@admin_bp.get("/api/admin/technician-assignments")
def admin_technician_assignments(admin_username: str):
    try:
        _require_admin(admin_username)
    except Exception as exc:
        _raise_http(exc)

    return get_technician_workload()


@admin_bp.post("/api/admin/prediction-logs/clear")
def clear_prediction_logs(req: ClearPredictionLogsRequest):
    try:
        _require_admin(req.admin_username)
    except Exception as exc:
        _raise_http(exc)

    count = clear_image_prediction_records(ids=req.ids)
    return {"status": "cleared", "deleted_count": count}
