"""
D Desk AI — Insights Router backed by Cloud Firestore.
"""

from datetime import datetime
from fastapi import APIRouter
from app.services.firestore_db import list_tickets
from app.services.llm_service import call_groq_with_metadata

insights_bp = APIRouter(tags=["Insights"])


@insights_bp.get("/api/insights")
async def get_insights(type: str = "summary"):
    """Generate LLM-powered admin insights."""
    tickets = list_tickets(include_deleted=False, limit=100)

    categories = ["Network", "Software", "Hardware", "Printer", "Other"]
    stats_text = (
        f"Total tickets: {len(tickets)}. "
        f"Resolved: {sum(1 for t in tickets if t.get('status') == 'resolved')}. "
        f"Auto-resolved: {sum(1 for t in tickets if t.get('auto_resolved'))}. "
        f"Categories: " + ", ".join(
            f"{cat}={sum(1 for t in tickets if t.get('category') == cat)}"
            for cat in categories
        )
    )

    prompts = {
        "summary":         f"Given these IT support stats: {stats_text}. Provide a 3-sentence executive summary.",
        "trends":          f"Given: {stats_text}. Identify the top 3 trends and what they indicate.",
        "recommendations": f"Given: {stats_text}. Give 3 concrete recommendations to improve IT support efficiency.",
        "report":          f"Given: {stats_text}. Produce a concise weekly IT operations report with totals, risks, and next actions.",
        "predictions":     f"Given historical data: {stats_text}. Predict next week's ticket volumes and recommend staffing.",
    }

    insight = await call_groq_with_metadata(
        prompt=prompts.get(type, prompts["summary"]),
        system="You are an expert IT operations analyst. Be concise, practical, and actionable.",
        max_tokens=1536,
    )

    return {
        "type": type,
        "insight": insight["text"],
        "source": insight["source"],
        "provider": insight["provider"],
        "provider_label": insight["provider_label"],
        "model": insight["model"],
        "groq_configured": insight["configured"],
        "fallback_reason": insight["error"],
        "generated_at": datetime.utcnow().isoformat(),
    }
