"""
Ticket management routes for live helpdesk workflows backed by Cloud Firestore.
"""

from __future__ import annotations

import calendar
import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import Config
from app.ml.predict import predict_ticket
from app.ml.spam import detect_spam_detailed
from app.services.duplicate_detector import find_duplicate, get_tfidf_json
from app.services.similarity import find_similar_resolved_tickets
from app.services.agentic_ai import (
    assign_technician_for_specialization,
    category_to_specialization,
    issue_to_specialization,
)
from app.services.groq_service import (
    call_groq_with_metadata,
    choose_final_priority,
    get_ticket_priority_advice,
    groq_classify_ticket,
    groq_verify_spam,
    groq_technician_ai_help,
    groq_agentic_ticket_analysis,
)
from app.services.firestore_db import (
    get_ticket,
    list_tickets,
    get_ticket_stats,
    create_ticket as db_create_ticket,
    update_ticket as db_update_ticket,
    soft_delete_ticket as db_soft_delete_ticket,
    restore_ticket as db_restore_ticket,
    add_ticket_reply as db_add_ticket_reply,
    list_ticket_replies as db_list_ticket_replies,
    get_user_by_username,
)

_tickets_logger = logging.getLogger("ddesk.tickets")

tickets_bp = APIRouter(tags=["Tickets"])

VALID_STATUSES = {"open", "in_progress", "resolved", "closed"}
VALID_RETENTION_UNITS = {"day", "days", "month", "months", "year", "years"}


class TicketCreateRequest(BaseModel):
    employee_id: str
    employee_name: Optional[str] = None
    location: Optional[str] = None
    description: str
    category: Optional[str] = None
    priority: Optional[str] = None
    source: str = "chat"
    predicted_issue: Optional[str] = None
    prediction_confidence: Optional[float] = None
    uploaded_image_path: Optional[str] = None
    image_prediction_id: Optional[str] = None


class TicketUpdateRequest(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    location: Optional[str] = None


class TicketDeleteRequest(BaseModel):
    admin_username: str
    retention_value: int = 30
    retention_unit: str = "days"


class TicketReplyCreateRequest(BaseModel):
    author_username: str
    author_name: Optional[str] = None
    author_role: str = "tech"
    message: str


class TicketAiHelpRequest(BaseModel):
    requester_username: Optional[str] = None
    requester_name: Optional[str] = None
    requester_role: str = "tech"
    extra_context: Optional[str] = None


class SimilarityRequest(BaseModel):
    description: str


def _http_400(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _normalize_status(status: str) -> str:
    normalized = (status or "").strip().lower().replace(" ", "_")
    if normalized == "progress":
        normalized = "in_progress"
    if normalized not in VALID_STATUSES:
        raise _http_400("Status must be one of: open, in_progress, resolved, closed.")
    return normalized


def _normalize_retention_unit(unit: str) -> str:
    normalized = (unit or "").strip().lower()
    if normalized not in VALID_RETENTION_UNITS:
        raise _http_400("Retention unit must be days, months, or years.")
    return normalized[:-1] if normalized.endswith("s") else normalized


def _require_admin(username: str) -> Dict[str, Any]:
    admin = get_user_by_username((username or "").strip().lower())
    if not admin or admin.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges are required for this action.")
    return admin


def _get_ticket_or_404(ticket_id: str, *, include_deleted: bool = True) -> Dict[str, Any]:
    ticket = get_ticket(ticket_id)
    if not ticket or (not include_deleted and ticket.get("deleted")):
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' was not found.")
    return ticket


def _build_ai_help_fallback(ticket: Dict[str, Any], extra_context: str = "") -> dict:
    combined_context = "\n".join(
        value for value in [ticket.get("description") or "", ticket.get("resolution") or "", extra_context or ""] if value
    )
    steps = [
        "Confirm the exact error message and whether the issue is affecting one user or multiple users.",
        "Repeat the issue once after a restart so you can separate a transient glitch from a persistent fault.",
        "Capture what changed recently, including updates, device moves, password resets, or network changes.",
    ]

    lower_text = combined_context.lower()
    if any(token in lower_text for token in ["wifi", "network", "internet", "vpn", "dns"]):
        steps = [
            "Check whether the user can reach both local resources and external sites to isolate the network path.",
            "Verify adapter, VPN, DNS, and switch-port status before escalating to infrastructure support.",
            "Capture gateway latency and signal strength if the device is wireless.",
        ]
    elif any(token in lower_text for token in ["printer", "print", "spooler", "paper", "toner"]):
        steps = [
            "Check power, paper tray, toner levels, and clear any physical paper blockage.",
            "Restart the print spooler service and verify the queue is not paused.",
            "Print a test page directly from the device console to rule out driver faults.",
        ]
    elif any(token in lower_text for token in ["screen", "display", "monitor", "flicker", "keyboard", "mouse", "battery", "laptop", "cable"]):
        steps = [
            "Inspect physical connectors, USB ports, and docking connections.",
            "Test with an alternate peripheral or external monitor to isolate hardware failure.",
            "Check device manager / system hardware report for disconnected peripherals.",
        ]

    return {
        "ticket_id": ticket.get("ticket_id"),
        "analysis": f"Diagnostic guidance for {ticket.get('category', 'general')} issue.",
        "recommended_steps": steps,
        "suggested_reply": (
            f"Hello {ticket.get('employee_name', 'there')}, I am reviewing your ticket regarding {ticket.get('description', 'the issue')}. "
            "I will be assisting you with troubleshooting shortly."
        ),
        "provider_label": "Local support guidance",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════

@tickets_bp.get("/api/stats")
def get_stats_route():
    """Retrieve overall ticket counts and operational stats."""
    return get_ticket_stats()


@tickets_bp.get("/api/tickets")
def list_tickets_route(
    employee_id: Optional[str] = None,
    assigned_to: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    include_deleted: bool = False,
    deleted_only: bool = False,
):
    tickets = list_tickets(
        employee_id=employee_id,
        assigned_to=assigned_to,
        status=status,
        limit=limit,
        include_deleted=include_deleted,
        deleted_only=deleted_only,
        include_replies=True,
    )
    return tickets


@tickets_bp.get("/api/tickets/deleted")
def list_deleted_tickets_route(admin_username: str, limit: int = 100):
    _require_admin(admin_username)
    return list_tickets(deleted_only=True, limit=limit, include_replies=True)


@tickets_bp.get("/api/tickets/{ticket_id}")
def get_ticket_route(ticket_id: str):
    ticket = _get_ticket_or_404(ticket_id, include_deleted=True)
    ticket["replies"] = db_list_ticket_replies(ticket_id)
    return ticket


@tickets_bp.post("/api/tickets")
async def create_ticket_route(req: TicketCreateRequest):
    description = (req.description or "").strip()
    if not description:
        raise HTTPException(
            status_code=400,
            detail="Ticket description cannot be empty. Please describe the IT issue you are facing.",
        )

    # 1. Spam & Phishing Gate
    spam_result = detect_spam_detailed(description)
    if spam_result.get("is_spam"):
        reason = spam_result.get("reason") or "Message flagged by helpdesk cybersecurity filter"
        raise HTTPException(
            status_code=422,
            detail=f"Ticket creation blocked: {reason}. Please describe a genuine IT problem (e.g. computer hardware, software, network, or printer issue).",
        )

    # 2. AI Takeover with Local ML Fallback
    ml_pred = predict_ticket(description)
    if req.category and req.priority:
        category = req.category
        priority = req.priority
        confidence = float(req.confidence or ml_pred.get("confidence", 0.95))
        pred_issue = req.predicted_issue or category.lower()
    else:
        ai_res = await groq_classify_ticket(description, fallback_prediction=ml_pred)
        category = req.category or ai_res["category"]
        priority = req.priority or ai_res["priority"]
        confidence = float(ai_res["confidence"])
        pred_issue = req.predicted_issue or ai_res.get("predicted_issue") or category.lower()

    # 3. Duplicate Detection
    is_duplicate, duplicate_of = find_duplicate(description, threshold=Config.DUPLICATE_THRESHOLD)

    # 4. Technician Assignment
    spec = category_to_specialization(category)
    assigned_tech = assign_technician_for_specialization(spec)
    assigned_name = assigned_tech.get("name") if assigned_tech else "Unassigned"

    # 5. TF-IDF vector for similarity
    tfidf_vec_json = get_tfidf_json(description)

    ticket_data = {
        "employee_id": req.employee_id,
        "employee_name": req.employee_name or req.employee_id,
        "location": req.location,
        "description": description,
        "category": category,
        "priority": priority,
        "status": "open",
        "source": req.source or "chat",
        "assigned_to": assigned_name,
        "predicted_issue": req.predicted_issue or category.lower(),
        "prediction_confidence": confidence,
        "technician_specialization": spec,
        "issue_resolution_status": "open",
        "auto_fix": None,
        "auto_resolved": False,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "tfidf_vec": tfidf_vec_json,
    }

    created = db_create_ticket(ticket_data)
    return created


@tickets_bp.patch("/api/tickets/{ticket_id}")
def update_ticket_route(ticket_id: str, req: TicketUpdateRequest):
    _get_ticket_or_404(ticket_id)
    updates = {}
    if req.status is not None:
        updates["status"] = _normalize_status(req.status)
    if req.assigned_to is not None:
        updates["assigned_to"] = req.assigned_to
    if req.resolution is not None:
        updates["resolution"] = req.resolution
    if req.location is not None:
        updates["location"] = req.location

    updated = db_update_ticket(ticket_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    updated["replies"] = db_list_ticket_replies(ticket_id)
    return updated


@tickets_bp.delete("/api/tickets/{ticket_id}")
def soft_delete_ticket_route(ticket_id: str, req: TicketDeleteRequest):
    _require_admin(req.admin_username)
    _get_ticket_or_404(ticket_id)
    return db_soft_delete_ticket(ticket_id, req.admin_username, retention_days=req.retention_value)


@tickets_bp.post("/api/tickets/{ticket_id}/restore")
def restore_ticket_route(ticket_id: str, req: dict):
    admin_username = req.get("admin_username", "")
    _require_admin(admin_username)
    _get_ticket_or_404(ticket_id)
    return db_restore_ticket(ticket_id)


@tickets_bp.post("/api/tickets/{ticket_id}/replies")
def add_ticket_reply_route(ticket_id: str, req: TicketReplyCreateRequest):
    ticket = _get_ticket_or_404(ticket_id, include_deleted=False)
    message = (req.message or "").strip()
    if not message:
        raise _http_400("Reply message cannot be empty.")

    author_username = (req.author_username or "").strip().lower()
    author = get_user_by_username(author_username)
    author_role = (author.get("role") if author else req.author_role).strip().lower()
    if author_role == "technician":
        author_role = "tech"

    if author_role not in {"employee", "tech"}:
        raise _http_400("Ticket chat is only between the employee and assigned technician.")

    author_name = (author.get("name") if author else req.author_name) or author_username

    # Security check: if tech replies, must match assigned_to
    if author_role == "tech" and ticket.get("assigned_to"):
        assigned = ticket.get("assigned_to", "").strip().lower()
        author_labels = {author_username, author_name.strip().lower()}
        if assigned not in author_labels and assigned != "unassigned":
            raise HTTPException(status_code=403, detail="Only the assigned technician can message this ticket.")

    reply = db_add_ticket_reply(
        ticket_id=ticket_id,
        author_username=author_username,
        author_name=author_name,
        author_role=author_role,
        message=message,
    )
    return reply


@tickets_bp.get("/api/tickets/{ticket_id}/replies")
def list_ticket_replies_route(ticket_id: str):
    _get_ticket_or_404(ticket_id)
    return db_list_ticket_replies(ticket_id)


@tickets_bp.get("/api/tickets/{ticket_id}/ai-assist")
@tickets_bp.post("/api/tickets/{ticket_id}/ai-assist")
@tickets_bp.post("/api/tickets/{ticket_id}/ai-help")
async def ticket_ai_help_route(ticket_id: str, req: Optional[TicketAiHelpRequest] = None):
    ticket = _get_ticket_or_404(ticket_id)
    similar = find_similar_resolved_tickets(ticket.get("description", ""))
    return await groq_agentic_ticket_analysis(
        ticket=ticket,
        similar_tickets=similar,
        requester_name=(req.requester_name if req else None) or (req.requester_username if req else None) or "",
    )


@tickets_bp.post("/api/tickets/similar")
@tickets_bp.post("/api/tickets/similarity")
def find_similar_tickets_route(req: SimilarityRequest):
    return find_similar_resolved_tickets(req.description)
