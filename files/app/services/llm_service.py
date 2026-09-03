"""
Backward-compatible wrapper around the Groq service helpers.
"""

from app.services.groq_service import call_groq, call_groq_with_metadata

__all__ = ["call_groq", "call_groq_with_metadata"]
