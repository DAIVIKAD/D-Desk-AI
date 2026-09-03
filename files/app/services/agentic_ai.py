"""
Agentic helpdesk helpers for image-led troubleshooting and technician routing.

These helpers are intentionally deterministic and local. The TensorFlow image
model still owns prediction; this module turns labels into beginner-safe
fix steps and assignment hints.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


TECH_SPECIALIZATIONS = ["hardware", "software", "networking", "printer_support", "general"]

SPECIALIZATION_ALIASES = {
    "hardware": "hardware",
    "hw": "hardware",
    "software": "software",
    "sw": "software",
    "network": "networking",
    "networking": "networking",
    "printer": "printer_support",
    "printer_support": "printer_support",
    "printer support": "printer_support",
    "general": "general",
    "other": "general",
}

ISSUE_SPECIALIZATION_MAP = {
    "screen_damage": "hardware",
    "screen_good": "general",
    "keyboard_issue": "hardware",
    "keyboard_good": "general",
    "mouse_issue": "hardware",
    "mouse_good": "general",
    "printer": "printer_support",
    "battery": "hardware",
    "cable_issue": "hardware",
    "overheating_issue": "hardware",
    "unavailable": "general",
    "other": "general",
}

CATEGORY_SPECIALIZATION_MAP = {
    "hardware": "hardware",
    "software": "software",
    "network": "networking",
    "networking": "networking",
    "printer": "printer_support",
    "other": "general",
}

ISSUE_CATEGORY_MAP = {
    "screen_damage": "Hardware",
    "screen_good": "Other",
    "keyboard_issue": "Hardware",
    "keyboard_good": "Other",
    "mouse_issue": "Hardware",
    "mouse_good": "Other",
    "printer": "Printer",
    "battery": "Hardware",
    "cable_issue": "Hardware",
    "overheating_issue": "Hardware",
    "unavailable": "Other",
    "other": "Other",
}

ISSUE_PRIORITY_MAP = {
    "screen_damage": "High",
    "battery": "High",
    "overheating_issue": "High",
    "printer": "Medium",
    "cable_issue": "Medium",
    "keyboard_issue": "Medium",
    "mouse_issue": "Medium",
}

FIX_SUGGESTIONS = {
    "screen_damage": [
        "Stop using the screen if it is cracked.",
        "Check for loose display cables.",
        "Request hardware replacement if damage is visible.",
    ],
    "keyboard_issue": [
        "Restart the computer once.",
        "Check if any key is stuck.",
        "Try another keyboard if available.",
    ],
    "mouse_issue": [
        "Reconnect the mouse cable or receiver.",
        "Try a different USB port.",
        "Replace the battery if it is wireless.",
    ],
    "printer": [
        "Check printer power and paper tray.",
        "Clear any stuck print jobs.",
        "Restart the printer and try again.",
    ],
    "battery": [
        "Connect the charger firmly.",
        "Check for battery warning messages.",
        "Avoid using a swollen or damaged battery.",
    ],
    "cable_issue": [
        "Check cable connection.",
        "Try reconnecting it.",
        "Replace damaged cable if needed.",
    ],
    "overheating_issue": [
        "Turn off the device for a few minutes.",
        "Keep air vents clear.",
        "Report it if the device stays hot.",
    ],
    "screen_good": [
        "No clear screen issue detected.",
        "Restart the device and check again.",
        "Create a ticket if the problem continues.",
    ],
    "keyboard_good": [
        "No clear keyboard issue detected.",
        "Test typing in another app.",
        "Create a ticket if keys still fail.",
    ],
    "mouse_good": [
        "No clear mouse issue detected.",
        "Try another USB port.",
        "Create a ticket if movement still fails.",
    ],
    "unavailable": [
        "Image model is not available right now.",
        "Describe the issue in text.",
        "Create a ticket if support is needed.",
    ],
    "other": [
        "Restart the affected device.",
        "Check power and connections.",
        "Create a ticket if it still does not work.",
    ],
}


def normalize_specialization(value: Optional[str]) -> str:
    """Return a stable technician specialization key."""
    normalized = (value or "").strip().lower().replace("-", "_")
    return SPECIALIZATION_ALIASES.get(normalized, "general")


def issue_to_specialization(issue_label: Optional[str]) -> str:
    return ISSUE_SPECIALIZATION_MAP.get((issue_label or "").strip().lower(), "general")


def category_to_specialization(category: Optional[str]) -> str:
    return CATEGORY_SPECIALIZATION_MAP.get((category or "").strip().lower(), "general")


def issue_to_category(issue_label: Optional[str]) -> str:
    return ISSUE_CATEGORY_MAP.get((issue_label or "").strip().lower(), "Other")


def issue_to_priority(issue_label: Optional[str], confidence: Optional[float] = None) -> str:
    label = (issue_label or "").strip().lower()
    if label in ISSUE_PRIORITY_MAP:
        return ISSUE_PRIORITY_MAP[label]
    if confidence is not None and confidence >= 0.85:
        return "Medium"
    return "Low"


def fix_suggestions_for_issue(issue_label: Optional[str]) -> List[str]:
    label = (issue_label or "").strip().lower()
    return FIX_SUGGESTIONS.get(label, FIX_SUGGESTIONS["other"])


def format_fix_suggestions(issue_label: Optional[str]) -> str:
    return "\n".join(f"- {step}" for step in fix_suggestions_for_issue(issue_label))


def humanize_issue(issue_label: Optional[str]) -> str:
    label = (issue_label or "other").strip().lower()
    return label.replace("_", " ").title()


def assign_technician_for_specialization(specialization: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Pick the least-loaded technician for a specialization using Firestore.
    """
    from app.services.firestore_db import assign_technician_by_specialization
    return assign_technician_by_specialization(specialization)
