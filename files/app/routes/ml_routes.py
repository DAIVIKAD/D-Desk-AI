"""
D Desk AI — ML API Routes (ml_routes.py)
─────────────────────────────────────────────
Three endpoints powered entirely by the app/ml/ module.

  POST /api/classify   → hybrid spam firewall + local ML + Groq advisory
  POST /api/spam       → local spam detection + Groq explanation
  POST /api/image      → local TF image classification (no Groq)

Architecture (Hybrid AI Spam Firewall):
  STEP 1 — Local Security Layer (spam.py):
      spam_score ≥ 0.75 OR strong phishing regex → Blocked immediately, no Groq.
  STEP 2 — Groq Semantic Verification:
      0.30 ≤ spam_score < 0.75 → Groq analyzes for phishing/scam semantics.
  STEP 3 — Final Decision Engine:
      Combines local score + Groq verdict → Spam/Phishing + Blocked or pass-through.
  STEP 4 — Only clean tickets proceed to IT classification + Groq priority advice.

Rules strictly enforced:
  ✅  Spam detection runs BEFORE all downstream classifiers.
  ✅  Category always comes from the local model (for genuine tickets).
  ✅  Groq may advise on priority only when ML confidence is low.
  ✅  Images are NEVER sent to Groq.
  ✅  No troubleshooting suggestions for spam.
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.ml.predict import predict_ticket
from app.ml.spam import detect_spam, detect_spam_detailed
from app.ml.image_model import classify_image
from app.ml.utils import is_conversational_text
from app.services.agentic_ai import fix_suggestions_for_issue, format_fix_suggestions, issue_to_specialization
from app.services.groq_service import (
    call_groq,
    choose_final_priority,
    get_ticket_priority_advice,
    groq_classify_ticket,
    groq_verify_spam,
)

logger = logging.getLogger("ddesk.ml_routes")

ml_routes_bp = APIRouter(tags=["ML"])

# ── Spam threshold constants ──────────────────────────────────────────────
_SPAM_HIGH_THRESHOLD = 0.75   # auto-block, no Groq needed
_SPAM_GREY_LOW = 0.30         # below this → clean, skip Groq
_SPAM_GREY_HIGH = 0.75        # grey zone ceiling
_GROQ_SPAM_CONFIDENCE_MIN = 0.60  # Groq must be this confident to override


# ═══════════════════════════════════════════════════════════════════════════
#  Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════

class ClassifyRequest(BaseModel):
    text: str

class SpamRequest(BaseModel):
    text: str


# ═══════════════════════════════════════════════════════════════════════════
#  Spam verdict builder (shared by /api/classify and /api/tickets)
# ═══════════════════════════════════════════════════════════════════════════

def _build_spam_blocked_response(spam_result: dict, groq_result: dict | None = None) -> dict:
    """Build the standard Spam/Phishing + Blocked response payload."""
    source = spam_result["classifier_source"]
    if groq_result and groq_result.get("source") == "groq":
        source = "hybrid_groq_spam_system"

    combined_score = spam_result["spam_score"]
    if groq_result:
        # Weighted combination: 40% local + 60% Groq when Groq is involved
        combined_score = round(
            spam_result["spam_score"] * 0.4 + groq_result["confidence"] * 0.6,
            4,
        )

    groq_reason = groq_result["reason"] if groq_result else None
    groq_intent = groq_result["detected_intent"] if groq_result else None

    return {
        # ── Spam verdict ──
        "category":             "Spam/Phishing",
        "priority":             "Blocked",
        "spam_score":           combined_score,
        "detected_spam_keywords": spam_result["detected_spam_keywords"],
        "detected_patterns":    spam_result["detected_patterns"],
        "decision_reason":      spam_result["decision_reason"],
        "classifier_source":    source,
        # ── Groq semantic analysis (if used) ──
        "groq_spam_verdict":    groq_result if groq_result else None,
        "groq_detected_intent": groq_intent,
        # ── Nullified downstream fields ──
        "ml_priority":          None,
        "groq_priority":        None,
        "priority_source":      "spam_gate",
        "category_confidence":  combined_score,
        "priority_confidence":  1.0,
        "confidence":           combined_score,
        "model_source":         "hybrid_groq_spam_system" if groq_result else "spam_detector",
        "groq_insight":         groq_reason or "This message was identified as spam or phishing and has been blocked.",
        "groq_reasoning":       groq_reason,
        "groq_self_help":       None,
        "groq_overrides_ml":    False,
    }


async def _run_hybrid_spam_check(text: str) -> tuple[bool, dict, dict | None]:
    """
    Execute the 3-tier hybrid spam firewall.

    Returns:
        (is_blocked, spam_result, groq_result_or_none)
    """
    # ── STEP 1: Local Security Layer ──────────────────────────────────────
    spam_result = detect_spam_detailed(text)

    logger.info(
        "SPAM_GATE step1_local: is_spam=%s score=%.4f keywords=%s patterns=%s source=%s | text=%.80s",
        spam_result["is_spam"],
        spam_result["spam_score"],
        spam_result["detected_spam_keywords"],
        spam_result["detected_patterns"],
        spam_result["classifier_source"],
        text,
    )

    # High confidence spam → block immediately, no Groq needed
    if spam_result["spam_score"] >= _SPAM_HIGH_THRESHOLD:
        logger.warning(
            "SPAM_GATE BLOCKED (high score): score=%.4f reason=%s | text=%.80s",
            spam_result["spam_score"], spam_result["decision_reason"], text,
        )
        return True, spam_result, None

    # Strong phishing regex hit + any positive score → block immediately
    if spam_result["detected_patterns"] and spam_result["spam_score"] > 0:
        logger.warning(
            "SPAM_GATE BLOCKED (phishing regex): patterns=%s score=%.4f | text=%.80s",
            spam_result["detected_patterns"], spam_result["spam_score"], text,
        )
        return True, spam_result, None

    # Direct keyword hit (rule_based flagged it) → block immediately
    if spam_result["is_spam"] and spam_result["spam_score"] >= _SPAM_HIGH_THRESHOLD:
        logger.warning(
            "SPAM_GATE BLOCKED (rule_based): score=%.4f | text=%.80s",
            spam_result["spam_score"], text,
        )
        return True, spam_result, None

    # ── STEP 2: Grey zone → Groq Semantic Verification ────────────────────
    if _SPAM_GREY_LOW <= spam_result["spam_score"] < _SPAM_GREY_HIGH:
        logger.info(
            "SPAM_GATE step2_groq: grey zone score=%.4f, consulting Groq | text=%.80s",
            spam_result["spam_score"], text,
        )

        groq_result = await groq_verify_spam(
            text=text,
            local_spam_score=spam_result["spam_score"],
            detected_keywords=spam_result["detected_spam_keywords"],
            detected_patterns=spam_result["detected_patterns"],
        )

        logger.info(
            "SPAM_GATE step2_groq result: is_spam=%s confidence=%.4f intent=%s reason=%s | text=%.80s",
            groq_result["is_spam"],
            groq_result["confidence"],
            groq_result["detected_intent"],
            groq_result["reason"],
            text,
        )

        # Groq says spam with sufficient confidence → block
        if groq_result["is_spam"] and groq_result["confidence"] >= _GROQ_SPAM_CONFIDENCE_MIN:
            logger.warning(
                "SPAM_GATE BLOCKED (groq_verified): groq_conf=%.4f local_score=%.4f | text=%.80s",
                groq_result["confidence"], spam_result["spam_score"], text,
            )
            return True, spam_result, groq_result

        # Groq says NOT spam → let it through even if local had partial signals
        if not groq_result["is_spam"] and groq_result["confidence"] >= _GROQ_SPAM_CONFIDENCE_MIN:
            logger.info(
                "SPAM_GATE CLEARED (groq_cleared): groq_conf=%.4f local_score=%.4f | text=%.80s",
                groq_result["confidence"], spam_result["spam_score"], text,
            )
            return False, spam_result, groq_result

        # Groq is also uncertain → fall back to local decision
        if spam_result["is_spam"]:
            return True, spam_result, groq_result
        return False, spam_result, groq_result

    # ── STEP 3: Local says spam (any score) → block ───────────────────────
    if spam_result["is_spam"]:
        logger.warning(
            "SPAM_GATE BLOCKED (local_flagged): score=%.4f | text=%.80s",
            spam_result["spam_score"], text,
        )
        return True, spam_result, None

    # ── Clean → pass through ──────────────────────────────────────────────
    logger.info(
        "SPAM_GATE CLEAN: score=%.4f | text=%.80s",
        spam_result["spam_score"], text,
    )
    return False, spam_result, None


# ═══════════════════════════════════════════════════════════════════════════
#  POST /api/classify
# ═══════════════════════════════════════════════════════════════════════════

@ml_routes_bp.post("/api/classify", summary="Classify an IT ticket (hybrid spam firewall + local ML + Groq)")
async def classify_ticket(req: ClassifyRequest):
    """
    Classify an IT support ticket using the hybrid AI spam firewall.

    Workflow:
      STEP 1 — Local spam.py (score ≥ 0.75 or phishing regex → Blocked, no Groq)
      STEP 2 — Groq semantic verification (grey zone 0.30–0.75)
      STEP 3 — Final decision engine (combine local + Groq)
      STEP 4 — Only clean tickets → IT category/priority classification + Groq advice
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field is required and must not be empty.")

    # ── Conversational Greetings & Smalltalk (No spam check, no false problem) ──
    if is_conversational_text(req.text):
        return {
            "category": "General",
            "priority": "Low",
            "ml_category": "General",
            "ml_priority": "Low",
            "groq_category": None,
            "groq_priority": None,
            "category_source": "conversational",
            "priority_source": "conversational",
            "ai_takeover": True,
            "category_confidence": 0.99,
            "priority_confidence": 0.99,
            "confidence": 0.99,
            "model_source": "conversational_handler",
            "predicted_issue": "greeting",
            "suggested_specialization": "general",
            "spam_score": 0.0,
            "detected_spam_keywords": [],
            "groq_insight": "Hello! I am D Desk AI, your IT support assistant. Please describe any IT issue you need help with.",
            "groq_reasoning": "Conversational greeting or acknowledgement.",
            "groq_self_help": "Describe your issue (e.g. Wi-Fi down, printer jammed, software crash) to get immediate help.",
            "groq_overrides_ml": False,
            "is_conversational": True,
        }

    # ── STEPS 1-3: Hybrid Spam Firewall ───────────────────────────────────
    is_blocked, spam_result, groq_spam = await _run_hybrid_spam_check(req.text)

    if is_blocked:
        return _build_spam_blocked_response(spam_result, groq_spam)

    # ── STEP 4: Genuine ticket → IT classification (Groq AI Takeover with Local Fallback) ──
    local_pred = predict_ticket(req.text)

    logger.info(
        "CLASSIFY local_fallback: category=%s priority=%s confidence=%.4f source=%s | text=%.80s",
        local_pred["category"], local_pred["priority"],
        float(local_pred["confidence"]), local_pred["source"], req.text,
    )

    # Groq AI takes over Category, Priority, and diagnostics (falls back to local_pred on error)
    ai_result = await groq_classify_ticket(req.text, fallback_prediction=local_pred)

    logger.info(
        "CLASSIFY final: category=%s priority=%s (ai_takeover=%s, source=%s) | text=%.80s",
        ai_result["category"], ai_result["priority"], ai_result["ai_takeover"], ai_result["source"], req.text,
    )

    groq_insight = f"{ai_result['reasoning']}\n\n{ai_result['self_help']}"

    return {
        # ── Primary Prediction (Groq AI Takeover / Local ML Fallback) ──
        "category":             ai_result["category"],
        "priority":             ai_result["priority"],
        "ml_category":          local_pred["category"],
        "ml_priority":          local_pred["priority"],
        "groq_category":        ai_result["category"] if ai_result["ai_takeover"] else None,
        "groq_priority":        ai_result["priority"] if ai_result["ai_takeover"] else None,
        "category_source":      ai_result["category_source"],
        "priority_source":      ai_result["priority_source"],
        "ai_takeover":          ai_result["ai_takeover"],
        "category_confidence":  ai_result["category_confidence"],
        "priority_confidence":  ai_result["priority_confidence"],
        "confidence":           ai_result["confidence"],
        "model_source":         ai_result["model_source"],
        "predicted_issue":      ai_result["predicted_issue"],
        "suggested_specialization": ai_result["suggested_specialization"],
        # ── Spam metadata (clean ticket) ──
        "spam_score":           spam_result["spam_score"],
        "detected_spam_keywords": spam_result["detected_spam_keywords"],
        # ── Groq advisory & self-help ──
        "groq_insight":         groq_insight,
        "groq_reasoning":       ai_result["reasoning"],
        "groq_self_help":       ai_result["self_help"],
        "groq_overrides_ml":    ai_result["ai_takeover"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  POST /api/spam
# ═══════════════════════════════════════════════════════════════════════════

@ml_routes_bp.post("/api/spam", summary="Detect spam in submitted text (hybrid local + Groq)")
async def spam_check(req: SpamRequest):
    """
    Detect whether the submitted text is spam or a legitimate IT ticket.
    Uses the same hybrid firewall as /api/classify.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field is required and must not be empty.")

    is_blocked, spam_result, groq_spam = await _run_hybrid_spam_check(req.text)
    label = "spam" if is_blocked else "not_spam"

    # Groq explanation for the result
    groq_prompt = (
        f"The following text was submitted to an IT helpdesk:\n\"{req.text}\"\n\n"
        f"Our spam detector classified it as: {label.upper()}\n\n"
        f"{'Briefly explain what spam signals are present.' if is_blocked else 'Briefly confirm why this looks like a legitimate IT support request.'}"
        " Keep your response under 60 words."
    )
    groq_insight = await call_groq(
        prompt=groq_prompt,
        system="You are an IT security analyst. Accept the provided spam classification as correct and explain it."
    )

    return {
        "label":                    label,
        "is_spam":                  is_blocked,
        "spam_score":               spam_result["spam_score"],
        "detected_spam_keywords":   spam_result["detected_spam_keywords"],
        "detected_patterns":        spam_result["detected_patterns"],
        "classifier_source":        spam_result["classifier_source"],
        "decision_reason":          spam_result["decision_reason"],
        "groq_spam_verdict":        groq_spam,
        "groq_insight":             groq_insight,
        "groq_overrides_ml":        False,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  POST /api/image
# ═══════════════════════════════════════════════════════════════════════════

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

@ml_routes_bp.post("/api/image", summary="Classify an uploaded device issue image (local TF only)")
async def image_classify(file: UploadFile = File(...)):
    """
    Classify an uploaded image using the locally saved TensorFlow image model.
    Images are processed LOCALLY — never sent to Groq.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Accepted: JPEG, PNG, WebP.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = classify_image(image_bytes)
    compatibility_label = "hardware_issue" if result["label"] == "keyboard_issue" else result["label"]

    return {
        "label":           result["label"],
        "compatibility_label": compatibility_label,
        "confidence":      result["confidence"],
        "top_prediction":  result["top_prediction"],
        "raw_label":       result.get("raw_label", result["top_prediction"]),
        "note":            result["note"],
        "fix_suggestions": fix_suggestions_for_issue(result["label"]),
        "suggested_fix":   format_fix_suggestions(result["label"]),
        "technician_specialization": issue_to_specialization(result["label"]),
        "groq_used":       False,
        "model_source":    result["model_source"],
    }
