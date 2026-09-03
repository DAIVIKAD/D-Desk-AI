"""
D Desk AI — Community Discussion Routes backed by Cloud Firestore.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ml.spam import detect_spam_detailed
from app.services.groq_service import call_groq
from app.services.firestore_db import (
    get_ticket,
    get_user_by_username,
    add_ticket_comment as db_add_ticket_comment,
    list_ticket_comments as db_list_ticket_comments,
    vote_ticket_comment as db_vote_ticket_comment,
    verify_ticket_comment as db_verify_ticket_comment,
    pin_ticket_comment as db_pin_ticket_comment,
    delete_ticket_comment as db_delete_ticket_comment,
)

logger = logging.getLogger("ddesk.comments")

comments_bp = APIRouter(tags=["Community Comments"])


class CommentCreate(BaseModel):
    employee_id: str
    employee_name: str = ""
    comment: str
    parent_comment_id: Optional[str] = None


class VoteRequest(BaseModel):
    employee_id: str


class VerifyRequest(BaseModel):
    username: str
    role: str          # "tech" or "admin"


class PinRequest(BaseModel):
    username: str
    role: str


class DeleteRequest(BaseModel):
    username: str
    role: str


@comments_bp.get("/api/tickets/{ticket_id}/comments")
def get_comments(ticket_id: str):
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    comments = db_list_ticket_comments(ticket_id)
    return {
        "ticket_id": ticket_id,
        "count": len(comments),
        "comments": comments,
    }


@comments_bp.post("/api/tickets/{ticket_id}/comments")
def add_comment(ticket_id: str, req: CommentCreate):
    if not req.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")

    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    # Spam gate
    spam_result = detect_spam_detailed(req.comment)
    if spam_result.get("is_spam"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Comment blocked — spam/phishing detected.",
                "spam_score": spam_result.get("spam_score"),
                "detected_spam_keywords": spam_result.get("detected_spam_keywords"),
                "decision_reason": spam_result.get("decision_reason"),
            },
        )

    employee_name = req.employee_name.strip()
    if not employee_name:
        user = get_user_by_username(req.employee_id)
        employee_name = user.get("name") if user else req.employee_id

    comment = db_add_ticket_comment(
        ticket_id=ticket_id,
        employee_id=req.employee_id,
        employee_name=employee_name,
        comment=req.comment.strip(),
        parent_comment_id=str(req.parent_comment_id) if req.parent_comment_id else None,
    )
    return {"message": "Comment added.", "comment": comment}


@comments_bp.post("/api/comments/{comment_id}/vote")
def vote_comment(comment_id: str, req: VoteRequest):
    updated = db_vote_ticket_comment(str(comment_id))
    if not updated:
        raise HTTPException(status_code=404, detail="Comment not found.")
    return {"message": "Vote recorded.", "comment": updated}


@comments_bp.post("/api/comments/{comment_id}/verify")
def verify_comment(comment_id: str, req: VerifyRequest):
    if req.role not in ("tech", "admin", "technician"):
        raise HTTPException(status_code=403, detail="Only technicians and admins can verify solutions.")

    updated = db_verify_ticket_comment(str(comment_id))
    if not updated:
        raise HTTPException(status_code=404, detail="Comment not found.")

    status = "verified" if updated.get("is_verified") else "unverified"
    return {"message": f"Comment {status}.", "comment": updated}


@comments_bp.post("/api/comments/{comment_id}/pin")
def pin_comment(comment_id: str, req: PinRequest):
    if req.role not in ("tech", "admin", "technician"):
        raise HTTPException(status_code=403, detail="Only technicians and admins can pin comments.")

    updated = db_pin_ticket_comment(str(comment_id))
    if not updated:
        raise HTTPException(status_code=404, detail="Comment not found.")

    status = "pinned" if updated.get("is_pinned") else "unpinned"
    return {"message": f"Comment {status}.", "comment": updated}


@comments_bp.delete("/api/comments/{comment_id}")
def delete_comment(comment_id: str, req: DeleteRequest):
    if req.role not in ("tech", "admin", "technician"):
        raise HTTPException(status_code=403, detail="Only technicians and admins can delete comments.")

    success = db_delete_ticket_comment(str(comment_id))
    return {"message": "Comment deleted."}


@comments_bp.get("/api/tickets/{ticket_id}/comments/summary")
async def comment_summary(ticket_id: str):
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    comments = db_list_ticket_comments(ticket_id)
    if not comments:
        return {
            "ticket_id": ticket_id,
            "summary": "No community comments yet.",
            "comment_count": 0,
            "has_verified": False,
        }

    verified = [c for c in comments if c.get("is_verified")]
    comment_lines = []
    for c in comments:
        prefix = "✓ VERIFIED" if c.get("is_verified") else f"({c.get('votes', 0)} votes)"
        comment_lines.append(f"- {c.get('employee_name', 'Employee')}: \"{c.get('comment', '')}\" {prefix}")

    prompt = (
        f"Ticket issue: \"{ticket.get('description', '')}\"\n\n"
        f"Community comments ({len(comments)} total):\n"
        + "\n".join(comment_lines[:20])
        + "\n\n"
        "Summarize the community suggestions in 1-2 sentences. "
        "Highlight the most helpful or verified solutions. "
        "Start with 'Community suggests:' and be actionable."
    )

    summary = await call_groq(
        prompt=prompt,
        system="You are an IT helpdesk AI summarizing employee troubleshooting suggestions. Be concise and actionable.",
    )

    return {
        "ticket_id": ticket_id,
        "summary": summary,
        "comment_count": len(comments),
        "has_verified": len(verified) > 0,
        "verified_count": len(verified),
    }
