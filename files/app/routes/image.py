"""
Image detection and troubleshooting routes backed by Cloud Firestore and in-memory ML inference.
Zero permanent image disk storage.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import Config
from app.ml.image_model import (
    SUPPORTED_IMAGE_LABELS,
    classify_image,
)
from app.services.agentic_ai import (
    assign_technician_for_specialization,
    fix_suggestions_for_issue,
    format_fix_suggestions,
    humanize_issue,
    issue_to_category,
    issue_to_priority,
    issue_to_specialization,
)
from app.services.duplicate_detector import find_duplicate, get_tfidf_json
from app.services.firestore_db import (
    create_image_prediction_record,
    get_image_prediction_record,
    update_image_prediction_record,
    create_ticket as db_create_ticket,
    get_ticket,
    get_user_by_username,
)

image_bp = APIRouter(tags=["Images"])
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class ImageFeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    correct_label: Optional[str] = None
    confirmed_by: Optional[str] = None


class ImageIssueResolutionRequest(BaseModel):
    prediction_id: str
    is_fixed: bool
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    location: Optional[str] = None
    confirmed_by: Optional[str] = None


@image_bp.post("/api/image-detect")
async def image_detect(
    file: UploadFile = File(...),
    created_by: Optional[str] = Form(default=None),
):
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Accepted: JPEG, PNG, WebP.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # In-memory inference using pretrained CNN / local heuristics
    prediction = classify_image(image_bytes)

    suggestions = fix_suggestions_for_issue(prediction["label"])
    suggested_fix = format_fix_suggestions(prediction["label"])
    technician_specialization = issue_to_specialization(prediction["label"])

    # Store prediction metadata only in Firestore (no persistent image file)
    record_data = {
        "filename": file.filename or "upload.jpg",
        "predicted_label": prediction["label"],
        "confidence": prediction["confidence"],
        "top_prediction": prediction["top_prediction"],
        "note": prediction["note"],
        "suggested_fix": suggested_fix,
        "created_by": (created_by or "").strip() or None,
        "technician_specialization": technician_specialization,
        "issue_resolution_status": "pending",
    }
    record = create_image_prediction_record(record_data)

    return {
        "prediction_id": record["id"],
        "prediction": prediction["label"],
        "confidence": prediction["confidence"],
        "top_prediction": prediction["top_prediction"],
        "raw_label": prediction.get("raw_label", prediction["top_prediction"]),
        "note": prediction["note"],
        "issue": prediction["label"],
        "issue_label": humanize_issue(prediction["label"]),
        "fix_suggestions": suggestions,
        "suggested_fix": suggested_fix,
        "technician_specialization": technician_specialization,
        "model_source": prediction["model_source"],
        "requires_confirmation": True,
        "requires_fix_confirmation": True,
    }


@image_bp.post("/api/image-resolution")
def image_resolution(req: ImageIssueResolutionRequest):
    record = get_image_prediction_record(str(req.prediction_id))
    if not record:
        raise HTTPException(status_code=404, detail="Image prediction record not found.")

    confirmed_by = (req.confirmed_by or req.employee_id or "").strip().lower() or None
    if req.is_fixed:
        updated = update_image_prediction_record(
            str(req.prediction_id),
            {
                "issue_resolution_status": "resolved",
                "confirmed_by": record.get("confirmed_by") or confirmed_by,
                "confirmed_at": datetime.utcnow().isoformat(),
            },
        )
        return {
            "status": "resolved",
            "ticket_created": False,
            "record": updated,
        }

    if record.get("ticket_id"):
        ticket = get_ticket(record["ticket_id"])
        if ticket:
            return {
                "status": "ticket_created",
                "ticket_created": True,
                "ticket": ticket,
                "record": record,
            }

    employee_id = (req.employee_id or record.get("created_by") or "").strip().lower()
    if not employee_id:
        raise HTTPException(status_code=400, detail="Employee ID is required to create a ticket.")

    employee = get_user_by_username(employee_id)
    employee_name = (req.employee_name or "").strip() or (employee.get("name") if employee else employee_id)
    issue_label = (record.get("confirmed_label") or record.get("predicted_label") or "other").strip().lower()
    confidence = float(record.get("confidence") or 0.0)
    category = issue_to_category(issue_label)
    priority = issue_to_priority(issue_label, confidence)
    technician_specialization = issue_to_specialization(issue_label)
    assigned_tech = assign_technician_for_specialization(technician_specialization)
    assigned_name = assigned_tech.get("name") if assigned_tech else "Unassigned"

    description = (
        f"Image-detected issue: {humanize_issue(issue_label)}.\n"
        f"Confidence: {round(confidence * 100)}%.\n"
        "User tried the suggested quick fix, but the issue is not resolved."
    )
    is_duplicate, duplicate_of = find_duplicate(description, threshold=Config.DUPLICATE_THRESHOLD)

    ticket_data = {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "location": (req.location or "").strip() or None,
        "description": description,
        "category": category,
        "priority": priority,
        "status": "open",
        "source": "image",
        "assigned_to": assigned_name,
        "predicted_issue": issue_label,
        "prediction_confidence": confidence,
        "technician_specialization": technician_specialization,
        "issue_resolution_status": "ticket_created",
        "auto_fix": record.get("suggested_fix") or format_fix_suggestions(issue_label),
        "auto_resolved": False,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "tfidf_vec": get_tfidf_json(description),
    }
    ticket = db_create_ticket(ticket_data)

    update_image_prediction_record(
        str(req.prediction_id),
        {
            "issue_resolution_status": "ticket_created",
            "ticket_id": ticket["ticket_id"],
            "assigned_to": ticket["assigned_to"],
            "technician_specialization": technician_specialization,
        },
    )

    return {
        "status": "ticket_created",
        "ticket_created": True,
        "ticket": ticket,
        "record": get_image_prediction_record(str(req.prediction_id)),
    }


@image_bp.post("/api/image-feedback")
def image_feedback(req: ImageFeedbackRequest):
    record = get_image_prediction_record(str(req.prediction_id))
    if not record:
        raise HTTPException(status_code=404, detail="Image prediction record not found.")

    correct_label = (req.correct_label or "").strip().lower()
    if not req.is_correct and correct_label not in SUPPORTED_IMAGE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image label '{correct_label}'. Must be one of {SUPPORTED_IMAGE_LABELS}.",
        )

    confirmed_label = record["predicted_label"] if req.is_correct else correct_label
    feedback_status = "confirmed" if req.is_correct else "corrected"

    updated = update_image_prediction_record(
        str(req.prediction_id),
        {
            "feedback_status": feedback_status,
            "confirmed_label": confirmed_label,
            "confirmed_by": (req.confirmed_by or "").strip().lower() or None,
            "confirmed_at": datetime.utcnow().isoformat(),
        },
    )

    return {
        "status": feedback_status,
        "prediction_id": req.prediction_id,
        "confirmed_label": confirmed_label,
        "record": updated,
    }
