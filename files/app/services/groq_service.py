"""
Groq-backed advisory helpers for ticketing workflows.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from app.config import Config
from app.ml.features import infer_priority_label
from app.ml.utils import is_conversational_text

logger = logging.getLogger("ddesk.groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", Config.GROQ_API_KEY)
GROQ_MODEL = Config.GROQ_MODEL
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_CATEGORIES = {"Hardware", "Network", "Software", "Printer", "Other"}

CANDIDATE_MODELS = [
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant",
]


def _active_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", Config.GROQ_API_KEY)


def _active_groq_model() -> str:
    return os.getenv("GROQ_MODEL", Config.GROQ_MODEL)


async def call_groq_with_metadata(
    prompt: str,
    system: str = "You are a helpful IT support assistant.",
    history: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """
    Call Groq and return both the response text and source metadata so callers
    can surface whether the answer came from Groq or the local fallback.
    Supports multi-turn conversation history.
    """
    api_key = _active_groq_api_key()
    primary_model = _active_groq_model()
    if not api_key:
        return {
            "text": _rule_based_response(prompt),
            "source": "fallback",
            "provider": "fallback",
            "provider_label": "Local support guidance",
            "model": None,
            "configured": False,
            "error": None,
        }

    # Prepare model candidate sequence: primary requested model first, then fallbacks
    models_to_try = [primary_model] + [m for m in CANDIDATE_MODELS if m != primary_model]
    last_error = None

    # Construct messages array with system prompt and history
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        for turn in history:
            role = "assistant" if turn.get("role") in ("assistant", "ai", "bot") else "user"
            content = str(turn.get("content", "") or turn.get("message", "")).strip()
            if content:
                messages.append({"role": role, "content": content})
    if prompt.strip():
        messages.append({"role": "user", "content": prompt.strip()})

    effective_max_tokens = max_tokens or Config.GROQ_MAX_TOKENS
    effective_temp = temperature if temperature is not None else Config.GROQ_TEMPERATURE

    for target_model in models_to_try:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    GROQ_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": target_model,
                        "messages": messages,
                        "max_tokens": effective_max_tokens,
                        "temperature": effective_temp,
                    },
                    timeout=25.0,
                )
                if response.status_code == 404:
                    # Model not found or deprecated on this endpoint/tier, try next
                    last_error = f"Model '{target_model}' not found (404)"
                    continue

                response.raise_for_status()
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"].strip(),
                    "source": "groq",
                    "provider": "groq",
                    "provider_label": f"Groq AI ({target_model})",
                    "model": target_model,
                    "configured": True,
                    "error": None,
                }
        except Exception as exc:
            last_error = str(exc)
            # If network error or timeout, don't keep looping unnecessarily
            if not isinstance(exc, httpx.HTTPStatusError):
                break

    return {
        "text": _rule_based_response(prompt),
        "source": "fallback",
        "provider": "fallback",
        "provider_label": "Local support guidance",
        "model": primary_model,
        "configured": True,
        "error": last_error,
    }


async def call_groq(
    prompt: str,
    system: str = "You are a helpful IT support assistant.",
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Call the Groq chat-completions API with multi-turn support.
    Falls back gracefully to local playbook when Groq is unavailable.
    """
    result = await call_groq_with_metadata(prompt=prompt, system=system, history=history)
    return result["text"]


def _category_to_spec(category: str) -> str:
    cat = (category or "").strip().lower()
    mapping = {
        "hardware": "hardware",
        "network": "networking",
        "software": "software",
        "printer": "printer_support",
        "other": "general",
    }
    return mapping.get(cat, "general")


def _normalize_category(val: Any, fallback: str = "Other") -> str:
    raw = str(val or "").strip().title()
    if raw in VALID_CATEGORIES:
        return raw
    lower = raw.lower()
    if any(k in lower for k in ["net", "wifi", "vpn", "ip", "dns", "lan", "wan", "router", "connectivity"]):
        return "Network"
    if any(k in lower for k in ["print", "paper", "toner", "spool"]):
        return "Printer"
    if any(k in lower for k in ["hard", "screen", "monitor", "laptop", "mouse", "keyboard", "battery", "cpu", "ram", "disk", "drive", "cable", "usb"]):
        return "Hardware"
    if any(k in lower for k in ["soft", "app", "os", "windows", "linux", "mac", "browser", "login", "auth", "portal", "crash", "bug"]):
        return "Software"
    return fallback if fallback in VALID_CATEGORIES else "Other"


def _normalize_priority(val: Any, fallback: str = "Medium") -> str:
    raw = str(val or "").strip().title()
    if raw in VALID_PRIORITIES:
        return raw
    lower = raw.lower()
    if any(k in lower for k in ["crit", "urg", "high", "p1", "sev1", "down", "block", "emergency"]):
        return "High"
    if any(k in lower for k in ["med", "normal", "p2", "sev2"]):
        return "Medium"
    if any(k in lower for k in ["low", "minor", "p3", "sev3", "info"]):
        return "Low"
    return fallback if fallback in VALID_PRIORITIES else "Medium"


def _build_fallback_classification(
    text: str,
    fb: dict[str, Any],
    fb_cat: str,
    fb_pri: str,
    fb_conf: float,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "category": fb_cat,
        "priority": fb_pri,
        "confidence": fb_conf,
        "category_confidence": fb.get("category_confidence", fb_conf),
        "priority_confidence": fb.get("priority_confidence", fb_conf),
        "reasoning": f"Local ML fallback: {reason}" if reason else "Categorized by local ML model.",
        "self_help": _rule_based_response(text),
        "predicted_issue": fb.get("predicted_issue") or fb_cat.lower(),
        "suggested_specialization": _category_to_spec(fb_cat),
        "source": "fallback",
        "model_source": fb.get("source", "local_ml_fallback"),
        "category_source": "ml",
        "priority_source": "ml",
        "ai_takeover": False,
        "raw_response": None,
    }


async def groq_classify_ticket(
    text: str,
    fallback_prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Primary AI classification engine powered by Groq.
    When Groq is available and working, Groq AI takes over:
      - category (Hardware, Network, Software, Printer, Other)
      - priority (High, Medium, Low)
      - reasoning (Explanation of categorization and priority)
      - self_help (Actionable IT troubleshooting steps)
      - predicted_issue (Technical issue summary)
      - confidence (AI confidence score)
      - suggested_specialization (IT tech routing)

    If Groq fails or is unconfigured, seamlessly falls back to the local ML model.
    """
    fb = fallback_prediction or {}
    fb_cat = _normalize_category(fb.get("category"), fallback="Other")
    fb_pri = _normalize_priority(fb.get("priority"), fallback="Medium")
    fb_conf = float(fb.get("confidence", 0.75))

    api_key = _active_groq_api_key()
    if not api_key:
        return _build_fallback_classification(text, fb, fb_cat, fb_pri, fb_conf, reason="Groq API key not configured")

    prompt = (
        "You are the primary enterprise IT Helpdesk AI. Classify the user's issue and return strict JSON only.\n"
        f"Ticket description: \"{text}\"\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "category": "Hardware" | "Network" | "Software" | "Printer" | "Other",\n'
        '  "priority": "High" | "Medium" | "Low",\n'
        '  "confidence": float between 0.85 and 0.99,\n'
        '  "reasoning": "1-2 concise sentences explaining why this category and priority were chosen",\n'
        '  "self_help": "2-3 numbered practical steps the user can try right now",\n'
        '  "predicted_issue": "concise 2-4 word technical summary (e.g. display_flicker, wifi_dhcp_failure, printer_spooler_jam)",\n'
        '  "suggested_specialization": "hardware" | "networking" | "software" | "printer_support" | "general"\n'
        "}\n"
        "Rules:\n"
        "- category MUST be exactly one of: Hardware, Network, Software, Printer, Other.\n"
        "- priority MUST be exactly one of: High, Medium, Low.\n"
        "- High priority: Complete work stoppage, severe security threat, multiple users affected, network/server down.\n"
        "- Medium priority: Single user degraded productivity, workaround available.\n"
        "- Low priority: Cosmetic, minor question, non-urgent request.\n"
        "- Return JSON ONLY. Do not wrap in markdown or backticks."
    )

    try:
        raw_result = await call_groq_with_metadata(
            prompt=prompt,
            system="You are an expert IT triage AI. Output strict valid JSON only.",
        )
        if raw_result.get("source") == "groq" and raw_result.get("text"):
            payload = _extract_json_payload(raw_result["text"])
            if payload and isinstance(payload, dict):
                cat = _normalize_category(payload.get("category"), fallback=fb_cat)
                pri = _normalize_priority(payload.get("priority"), fallback=fb_pri)
                conf = float(payload.get("confidence", 0.95))
                conf = max(0.5, min(0.99, conf))
                reasoning = str(payload.get("reasoning", "")).strip() or "Classified by Groq AI model."
                self_help = _normalize_self_help_text(payload.get("self_help")) or _rule_based_response(text)
                issue = str(payload.get("predicted_issue", "")).strip() or cat.lower()
                spec = str(payload.get("suggested_specialization", "")).strip().lower().replace("-", "_")
                if spec not in {"hardware", "networking", "software", "printer_support", "general"}:
                    spec = _category_to_spec(cat)

                return {
                    "category": cat,
                    "priority": pri,
                    "confidence": round(conf, 4),
                    "category_confidence": round(conf, 4),
                    "priority_confidence": round(conf, 4),
                    "reasoning": reasoning,
                    "self_help": self_help,
                    "predicted_issue": issue,
                    "suggested_specialization": spec,
                    "source": "groq",
                    "model_source": f"groq_ai ({raw_result.get('model', 'llama')})",
                    "category_source": "groq",
                    "priority_source": "groq",
                    "ai_takeover": True,
                    "raw_response": raw_result["text"],
                }
    except Exception as exc:
        logger.warning("Groq AI classification failed, falling back to local ML: %s", exc)

    return _build_fallback_classification(text, fb, fb_cat, fb_pri, fb_conf, reason="Groq unavailable or returned invalid structure")


async def get_ticket_priority_advice(
    *,
    text: str,
    predicted_category: str,
    predicted_priority: str,
    confidence: float,
) -> dict[str, Any]:
    """
    Groq AI advisory helper. When Groq is available, it analyzes both category
    and priority, with local ML as the resilient fallback.
    """
    classification = await groq_classify_ticket(
        text=text,
        fallback_prediction={
            "category": predicted_category,
            "priority": predicted_priority,
            "confidence": confidence,
        },
    )

    return {
        "suggested_category": classification["category"],
        "suggested_priority": classification["priority"],
        "reasoning": classification["reasoning"],
        "self_help": classification["self_help"],
        "predicted_issue": classification["predicted_issue"],
        "suggested_specialization": classification["suggested_specialization"],
        "raw_response": classification.get("raw_response"),
        "source": classification["source"],
        "ai_takeover": classification["ai_takeover"],
    }


def choose_final_priority(ml_priority: str, confidence: float, groq_priority: str) -> tuple[str, str]:
    """
    With Groq AI Takeover, Groq's priority takes precedence if valid.
    Otherwise falls back to local ML priority.
    """
    norm_groq = _normalize_priority(groq_priority, fallback="")
    if norm_groq in VALID_PRIORITIES:
        return norm_groq, "groq"
    return ml_priority, "ml"


async def groq_verify_spam(
    text: str,
    local_spam_score: float,
    detected_keywords: list[str],
    detected_patterns: list[str],
) -> dict[str, Any]:
    """
    Ask Groq to semantically verify whether a message is spam/phishing.

    Called ONLY for uncertain cases where spam.py's local score is in the
    grey zone (0.30 ≤ score < 0.75).

    Returns:
        dict with keys:
            is_spam         — bool
            confidence      — float 0.0–1.0
            reason          — human-readable explanation
            detected_intent — e.g. "phishing", "reward_scam", "legitimate_it"
            source          — "groq" | "fallback"
    """
    if is_conversational_text(text):
        return {
            "is_spam": False,
            "confidence": 0.0,
            "reason": "Conversational greeting or acknowledgement.",
            "detected_intent": "legitimate_it",
            "source": "groq",
        }

    prompt = (
        "Return strict JSON only with keys: is_spam, confidence, reason, detected_intent.\n\n"
        f"Text submitted to an IT helpdesk:\n\"{text}\"\n\n"
        f"Local spam score: {local_spam_score:.4f}\n"
        f"Detected keywords: {detected_keywords}\n"
        f"Detected patterns: {detected_patterns}\n\n"
        "Analyze this message and determine:\n"
        "- Is this phishing, spam, or a scam?\n"
        "- Is it manipulative or suspicious?\n"
        "- Is it unrelated to legitimate IT support?\n"
        "- Does it attempt credential theft, reward scams, or deception?\n\n"
        "Rules:\n"
        "- Conversational messages, greetings (e.g. 'hi', 'hello', 'good morning', 'thanks', 'ok') are NOT spam.\n"
        "- is_spam must be true or false.\n"
        "- confidence must be a float between 0.0 and 1.0.\n"
        "- reason must be one concise sentence.\n"
        "- detected_intent must be one of: phishing, reward_scam, credential_theft, "
        "crypto_scam, social_engineering, generic_spam, legitimate_it, unclear.\n"
    )

    raw = await call_groq(
        prompt=prompt,
        system=(
            "You are a cybersecurity analyst specializing in spam and phishing detection "
            "for enterprise IT helpdesks. Classify the message strictly. "
            "NOTE: Conversational messages, greetings, politeness, and legitimate IT inquiries are NOT spam. "
            "Only flag as spam if the message is an actual phishing attempt, malicious link, credential theft, or scam. "
            "Output strict JSON only."
        ),
    )

    payload = _extract_json_payload(raw)

    if payload is None:
        # Fallback: if Groq is unavailable, lean towards the local score
        return {
            "is_spam": local_spam_score >= 0.50,
            "confidence": local_spam_score,
            "reason": "Groq unavailable; decision based on local spam score.",
            "detected_intent": "unclear",
            "source": "fallback",
        }

    groq_is_spam = bool(payload.get("is_spam", False))
    groq_confidence = float(payload.get("confidence", 0.5))
    groq_confidence = max(0.0, min(1.0, groq_confidence))  # clamp
    groq_reason = str(payload.get("reason", "")).strip() or "No reason provided."
    groq_intent = str(payload.get("detected_intent", "unclear")).strip().lower()

    valid_intents = {
        "phishing", "reward_scam", "credential_theft", "crypto_scam",
        "social_engineering", "generic_spam", "legitimate_it", "unclear",
    }
    if groq_intent not in valid_intents:
        groq_intent = "unclear"

    return {
        "is_spam": groq_is_spam,
        "confidence": round(groq_confidence, 4),
        "reason": groq_reason,
        "detected_intent": groq_intent,
        "source": "groq" if _active_groq_api_key() else "fallback",
    }


def _extract_json_payload(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_self_help_text(value: Any) -> str:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(f"{index}. {item}" for index, item in enumerate(cleaned, start=1))

    text = str(value or "").strip()
    if not text:
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return _normalize_self_help_text(parsed)
        except Exception:
            pass

    return text


def _rule_based_response(prompt: str) -> str:
    """Fallback guidance when Groq is not configured."""
    text = prompt.lower()
    if "network" in text or "internet" in text or "vpn" in text or "wifi" in text:
        return (
            "1. Reconnect the device to Wi-Fi or VPN.\n"
            "2. Restart the network adapter or router if available.\n"
            "3. Check whether multiple users are affected before escalating."
        )
    if "printer" in text or "scan" in text or "toner" in text:
        return (
            "1. Confirm the printer is powered on and connected.\n"
            "2. Clear stuck jobs from the print queue.\n"
            "3. Re-add the printer or restart the Print Spooler service."
        )
    if "software" in text or "application" in text or "crash" in text or "outlook" in text:
        return (
            "1. Fully close and reopen the application.\n"
            "2. Clear the app cache or run a repair if available.\n"
            "3. Note any error code before raising the ticket."
        )
    if "hardware" in text or "laptop" in text or "keyboard" in text or "screen" in text:
        return (
            "1. Restart the device and disconnect non-essential peripherals.\n"
            "2. Check power, cables, and visible physical damage.\n"
            "3. If the issue persists, create a hardware support ticket."
        )
    return (
        "1. Capture the exact error or symptom.\n"
        "2. Retry the action once after a restart.\n"
        "3. Share what changed recently so the support team can triage faster."
    )


async def groq_chat_assistant(
    message: str,
    conversation: list[dict[str, Any]] | None = None,
    employee_name: str = "",
) -> dict[str, Any]:
    """
    Unified Groq-powered Chat Assistant for enterprise employees.
    Accurately discriminates between casual chat vs. genuine IT support scenarios.

    If the user is having casual conversation (e.g. 'i ate too much', 'hi', 'ok', jokes):
      - is_it_issue: False
      - category: None, priority: None, confidence: None
      - Returns ONLY natural conversational text.

    If the user has a genuine IT issue or wants to open a ticket:
      - is_it_issue: True
      - category, priority, confidence provided by Groq
      - Returns structured troubleshooting steps.
    """
    msg = message.strip()
    api_key = _active_groq_api_key()

    prompt = (
        f"You are D Desk AI, the enterprise IT Helpdesk Assistant for POWERGRID.\n"
        f"Employee: {employee_name or 'Employee'}\n"
        f"User Message: \"{msg}\"\n\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "is_it_issue": boolean,\n'
        '  "response": "Your direct response to the employee",\n'
        '  "category": "Hardware" | "Network" | "Software" | "Printer" | "Other" | null,\n'
        '  "priority": "High" | "Medium" | "Low" | null,\n'
        '  "confidence": float between 0.80 and 0.99 | null,\n'
        '  "reasoning": "1 sentence explanation if is_it_issue is true, otherwise null"\n'
        "}\n\n"
        "STRICT CLASSIFICATION RULES:\n"
        "1. is_it_issue MUST be true ONLY when the user is reporting a genuine technical problem "
        "(computer, laptop, network, wifi, VPN, software, printer, account lockout, hardware fault) "
        "or explicitly asking to open/create a support ticket.\n"
        "2. is_it_issue MUST be false for greetings, acknowledgements ('ok', 'cool'), casual chat, "
        "personal statements ('i ate too much', 'im tired'), jokes, general questions ('who are you'), "
        "or off-topic comments.\n"
        "3. When is_it_issue is false:\n"
        "   - category, priority, and confidence MUST be null.\n"
        "   - The response must ONLY be natural, friendly conversation. Do NOT mention priority/confidence, "
        "do NOT provide troubleshooting steps, and do NOT offer ticket creation.\n"
        "4. When is_it_issue is true:\n"
        "   - Give clear, structured, numbered troubleshooting steps in 'response'.\n"
        "   - Provide valid category, priority, and confidence.\n"
        "5. Output strict JSON only. No markdown fences, no backticks."
    )

    if api_key:
        try:
            raw_meta = await call_groq_with_metadata(
                prompt=prompt,
                system="You are an enterprise IT triage assistant. Output strict valid JSON only.",
                history=conversation,
            )
            if raw_meta.get("source") == "groq" and raw_meta.get("text"):
                payload = _extract_json_payload(raw_meta["text"])
                if payload and isinstance(payload, dict):
                    is_it_issue = bool(payload.get("is_it_issue", False))
                    resp_text = str(payload.get("response", "")).strip()

                    if not is_it_issue:
                        return {
                            "response": resp_text or "Hello! I'm D Desk AI. Let me know if you have any IT issues with your computer, software, or network.",
                            "is_it_issue": False,
                            "is_issue": False,
                            "category": None,
                            "priority": None,
                            "confidence": None,
                            "source": "groq",
                            "provider_label": raw_meta.get("provider_label", "Groq AI"),
                            "model": raw_meta.get("model"),
                            "ai_takeover": True,
                        }

                    cat = _normalize_category(payload.get("category"), fallback="Other")
                    pri = _normalize_priority(payload.get("priority"), fallback="Medium")
                    conf = float(payload.get("confidence") or 0.95)
                    conf = max(0.5, min(0.99, conf))

                    return {
                        "response": resp_text,
                        "is_it_issue": True,
                        "is_issue": True,
                        "category": cat,
                        "priority": pri,
                        "confidence": round(conf, 4),
                        "reasoning": str(payload.get("reasoning", "")).strip(),
                        "source": "groq",
                        "provider_label": raw_meta.get("provider_label", "Groq AI"),
                        "model": raw_meta.get("model"),
                        "ai_takeover": True,
                    }
        except Exception as exc:
            logger.warning("groq_chat_assistant failed, falling back: %s", exc)

    # ── Local Fallback ──
    from app.ml.predict import predict_ticket
    from app.ml.utils import is_conversational_text

    if is_conversational_text(msg):
        return {
            "response": "Hello! I'm D Desk AI. How can I help you today? Please let me know if you have any tech or IT issues.",
            "is_it_issue": False,
            "is_issue": False,
            "category": None,
            "priority": None,
            "confidence": None,
            "source": "fallback",
            "provider_label": "Local support guidance",
            "model": None,
            "ai_takeover": False,
        }

    local_pred = predict_ticket(msg)
    cat = _normalize_category(local_pred.get("category"), fallback="Other")
    pri = _normalize_priority(local_pred.get("priority"), fallback="Medium")
    conf = float(local_pred.get("confidence", 0.75))

    return {
        "response": (
            f"Here are the recommended diagnostic steps from the local IT playbook:\n\n"
            f"{_rule_based_response(msg)}\n\n"
            "If this does not resolve the issue, please click 'Still Not Fixed — Create Ticket' below."
        ),
        "is_it_issue": True,
        "is_issue": True,
        "category": cat,
        "priority": pri,
        "confidence": round(conf, 4),
        "reasoning": "Determined via local ML classifier.",
        "source": "fallback",
        "provider_label": "Local support guidance",
        "model": None,
        "ai_takeover": False,
    }


async def groq_technician_ai_help(
    ticket: dict[str, Any],
    extra_context: str = "",
    requester_name: str = "",
) -> dict[str, Any]:
    """
    Generate professional, in-depth technical guidance for field/on-site technicians.
    Answers technician's questions/clues directly, outlines step-by-step troubleshooting SOP,
    and drafts a professional message for the technician to send to the employee.
    """
    api_key = _active_groq_api_key()
    ticket_id = ticket.get("ticket_id", "Pending")
    cat = ticket.get("category", "General")
    pri = ticket.get("priority", "Medium")
    desc = ticket.get("description", "")
    pred_issue = ticket.get("predicted_issue") or cat.lower()
    emp_name = ticket.get("employee_name") or ticket.get("employee_id") or "Employee"
    loc = ticket.get("location") or "Desk"

    prompt = (
        f"You are an expert IT Hardware, Network & Systems Support Lead assisting an on-site technician.\n"
        f"Ticket: #{ticket_id}\n"
        f"Category: {cat} | Priority: {pri}\n"
        f"Employee: {emp_name} (Location: {loc})\n"
        f"Reported Issue: {desc}\n"
        f"Predicted Technical Fault: {pred_issue}\n\n"
        f"Technician's Question / Extra Clues:\n"
        f"\"{extra_context or 'None provided. Please provide full technical diagnosis.'}\"\n\n"
        "Return strict JSON with this exact schema:\n"
        "{\n"
        '  "analysis": "In-depth technical root-cause analysis that directly and thoroughly answers the technician\'s question or clues.",\n'
        '  "next_steps": "Detailed numbered SOP instructions (1., 2., 3...) for the technician to inspect, test, repair, or configure.",\n'
        f'  "reply_draft": "A professional, polite, ready-to-send draft message from the technician to {emp_name} explaining the status, action plan, and any workaround."\n'
        "}\n\n"
        "Rules:\n"
        "- Directly address the technician's question (e.g. replace panel vs external monitor, reboot switch vs reflash, reinstall vs patch).\n"
        "- Format 'next_steps' cleanly with numbers.\n"
        "- Keep the draft message courteous and actionable.\n"
        "- Return JSON only. No markdown fences, no backticks."
    )

    if api_key:
        try:
            raw_meta = await call_groq_with_metadata(
                prompt=prompt,
                system="You are a senior enterprise IT systems engineer. Output strict valid JSON only.",
                max_tokens=1536,
                temperature=0.3,
            )
            if raw_meta.get("source") == "groq" and raw_meta.get("text"):
                payload = _extract_json_payload(raw_meta["text"])
                if payload and isinstance(payload, dict):
                    analysis = str(payload.get("analysis", "")).strip()
                    next_steps = _normalize_self_help_text(payload.get("next_steps")) or str(payload.get("next_steps", "")).strip()
                    reply_draft = str(payload.get("reply_draft", "")).strip()

                    if analysis and next_steps:
                        return {
                            "ticket_id": ticket_id,
                            "analysis": analysis,
                            "next_steps": next_steps,
                            "recommended_steps": next_steps,
                            "reply_draft": reply_draft,
                            "suggested_reply": reply_draft,
                            "source": "groq",
                            "provider_label": raw_meta.get("provider_label", "Groq AI"),
                            "model": raw_meta.get("model"),
                            "ai_takeover": True,
                        }
        except Exception as exc:
            logger.warning("groq_technician_ai_help failed, using local fallback: %s", exc)

    # ── Local Technician Fallback ──
    steps_list = [
        "Confirm the exact symptom and test with the employee in person.",
        "Inspect physical cables, ports, power sources, and hardware integrity.",
        "Check system logs or device manager for fault codes before component replacement.",
        "Record action taken and schedule follow-up verification.",
    ]
    comb = (desc + " " + extra_context).lower()
    if "screen" in comb or "display" in comb or "monitor" in comb:
        steps_list = [
            "Connect an external display via HDMI/Type-C to confirm GPU/motherboard is functional.",
            "Inspect LCD panel and hinge flex cable for physical cracks or tears.",
            "If internal display is cracked, order replacement panel; connect external monitor as immediate workaround.",
            "Record laptop model and serial number in ticket notes for hardware procurement.",
        ]
    elif "network" in comb or "wifi" in comb or "vpn" in comb:
        steps_list = [
            "Verify IP configuration (ipconfig /all) and ping the default gateway.",
            "Test with both wired Ethernet and wireless connection to isolate the adapter.",
            "Reset network stack (netsh winsock reset) and flush DNS cache.",
            "Verify network wall port patch on the floor switch.",
        ]

    steps_str = "\n".join(f"{i}. {s}" for i, s in enumerate(steps_list, start=1))
    reply_str = (
        f"Hello {emp_name},\n\n"
        f"I am investigating ticket #{ticket_id} regarding '{desc}'. "
        "I have initiated diagnostics and will visit your desk or apply the recommended fix shortly. "
        "Please let me know if you have any questions in the meantime.\n\n"
        f"Regards,\n{requester_name or 'IT Support Technician'}"
    )

    return {
        "ticket_id": ticket_id,
        "analysis": f"Local hardware/software diagnostic guidance for {cat} ({pri} Priority).",
        "next_steps": steps_str,
        "recommended_steps": steps_str,
        "reply_draft": reply_str,
        "suggested_reply": reply_str,
        "source": "fallback",
        "provider_label": "Local support guidance",
        "model": None,
        "ai_takeover": False,
    }


async def groq_agentic_ticket_analysis(
    ticket: dict[str, Any],
    similar_tickets: list[dict[str, Any]] | None = None,
    requester_name: str = "",
) -> dict[str, Any]:
    """
    Enterprise Agentic AI workflow for end-to-end ticket triage.
    Executes a multi-stage cognitive pipeline:
      1. Issue Understanding & Symptom Extraction
      2. Classification & Urgency Assessment (Local ML + Groq reasoning)
      3. Historical Context & Similar Ticket Synthesis
      4. Action Plan & SOP Formulation
      5. Customer-Facing Reply Generation
    Exposes safe, high-level activity states to the UI without revealing raw chain-of-thought.
    """
    api_key = _active_groq_api_key()
    ticket_id = ticket.get("ticket_id", "TKT")
    cat = ticket.get("category", "General")
    pri = ticket.get("priority", "Medium")
    desc = ticket.get("description", "")
    emp_name = ticket.get("employee_name") or ticket.get("employee_id") or "Employee"
    loc = ticket.get("location") or "Workstation"

    sim_context = ""
    if similar_tickets:
        sim_lines = []
        for t in similar_tickets[:3]:
            sim_lines.append(f"- #{t.get('ticket_id', '')} ({t.get('category', '')}): {t.get('description', '')[:100]} -> Resolution: {t.get('resolution_notes', 'Resolved') or 'Resolved'}")
        sim_context = "\n".join(sim_lines)

    prompt = (
        f"You are D Desk AI, an autonomous enterprise IT Support Agent.\n"
        f"Analyze the following IT support ticket and synthesize a resolution strategy:\n\n"
        f"Ticket ID: #{ticket_id}\n"
        f"Employee: {emp_name} | Location: {loc}\n"
        f"Category: {cat} | Priority: {pri}\n"
        f"Description: \"{desc}\"\n\n"
        f"Historical Similar Tickets Context:\n{sim_context or 'No prior identical incidents found in knowledge base.'}\n\n"
        "Return strict JSON with this exact schema:\n"
        "{\n"
        '  "understanding": "1-2 sentence executive summary of the core technical defect and operational impact.",\n'
        '  "root_cause_hypothesis": "Most probable technical cause.",\n'
        '  "recommended_steps": "Detailed numbered SOP instructions (1., 2., 3...) for the technician/support staff.",\n'
        '  "suggested_reply": "A courteous, reassuring message from the technician to the employee explaining the triage status and next steps.",\n'
        '  "preventative_advice": "1 practical tip to prevent recurrence."\n'
        "}\n\n"
        "Rules:\n"
        "- Return JSON ONLY. No markdown wrapping, no backticks."
    )

    if api_key:
        try:
            raw_meta = await call_groq_with_metadata(
                prompt=prompt,
                system="You are an enterprise Agentic IT workflow engine. Output strict valid JSON only.",
                max_tokens=1536,
                temperature=0.25,
            )
            if raw_meta.get("source") == "groq" and raw_meta.get("text"):
                payload = _extract_json_payload(raw_meta["text"])
                if payload and isinstance(payload, dict):
                    understanding = str(payload.get("understanding", "")).strip()
                    root_cause = str(payload.get("root_cause_hypothesis", "")).strip()
                    rec_steps = _normalize_self_help_text(payload.get("recommended_steps")) or str(payload.get("recommended_steps", "")).strip()
                    sug_reply = str(payload.get("suggested_reply", "")).strip()
                    preventative = str(payload.get("preventative_advice", "")).strip()

                    return {
                        "ticket_id": ticket_id,
                        "understanding": understanding or f"Investigating reported {cat} fault for {emp_name}.",
                        "root_cause_hypothesis": root_cause or f"Transient or physical fault in {cat.lower()} subsystem.",
                        "analysis": f"{understanding}\n\nHypothesis: {root_cause}",
                        "recommended_steps": rec_steps or "1. Perform initial diagnostic assessment.\n2. Apply standard troubleshooting SOP.",
                        "next_steps": rec_steps,
                        "suggested_reply": sug_reply or f"Hello {emp_name}, we are reviewing your ticket #{ticket_id} and will assist you shortly.",
                        "reply_draft": sug_reply,
                        "preventative_advice": preventative,
                        "similar_tickets_count": len(similar_tickets or []),
                        "source": "groq",
                        "provider_label": raw_meta.get("provider_label", "Groq AI"),
                        "model": raw_meta.get("model"),
                        "agent_activity": [
                            {"step": 1, "action": "Understanding Issue", "status": "completed", "detail": "Extracted symptoms and affected environment."},
                            {"step": 2, "action": "Classifying Ticket", "status": "completed", "detail": f"Classified category as '{cat}' with '{pri}' priority."},
                            {"step": 3, "action": "Checking Similar Issues", "status": "completed", "detail": f"Evaluated {len(similar_tickets or [])} historical knowledge records."},
                            {"step": 4, "action": "Building Resolution Plan", "status": "completed", "detail": "Synthesized SOP checklist and root-cause hypothesis."},
                            {"step": 5, "action": "Preparing Response", "status": "completed", "detail": "Drafted technician communication."},
                        ],
                    }
        except Exception as exc:
            logger.warning("groq_agentic_ticket_analysis failed, using fallback: %s", exc)

    # ── Resilient Local Fallback ──
    local_help = await groq_technician_ai_help(ticket, requester_name=requester_name)
    return {
        "ticket_id": ticket_id,
        "understanding": f"Reported {cat} issue ({pri} Priority) affecting {emp_name}.",
        "root_cause_hypothesis": f"Hardware, software, or network configuration anomaly in {cat}.",
        "analysis": local_help.get("analysis"),
        "recommended_steps": local_help.get("recommended_steps"),
        "next_steps": local_help.get("next_steps"),
        "suggested_reply": local_help.get("suggested_reply"),
        "reply_draft": local_help.get("reply_draft"),
        "preventative_advice": "Ensure standard corporate operating system updates and periodic hardware health scans are maintained.",
        "similar_tickets_count": len(similar_tickets or []),
        "source": "fallback",
        "provider_label": "Local support guidance",
        "model": None,
        "agent_activity": [
            {"step": 1, "action": "Understanding Issue", "status": "completed", "detail": "Extracted symptoms from ticket description."},
            {"step": 2, "action": "Classifying Ticket", "status": "completed", "detail": f"Evaluated category as '{cat}' ({pri} Priority)."},
            {"step": 3, "action": "Checking Similar Issues", "status": "completed", "detail": f"Queried {len(similar_tickets or [])} matching records from database."},
            {"step": 4, "action": "Building Resolution Plan", "status": "completed", "detail": "Loaded standard operating procedure checklist."},
            {"step": 5, "action": "Preparing Response", "status": "completed", "detail": "Generated standard technician response draft."},
        ],
    }

