from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.groq_service import groq_chat_assistant

chat_bp = APIRouter(tags=["Chat"])


class ChatMessage(BaseModel):
    employee_id: str = ""
    employee_name: str = ""
    message: str
    conversation: List[Dict[str, Any]] = []
    location: Optional[str] = None


@chat_bp.post("/api/chat")
async def chat(req: ChatMessage):
    """
    Enterprise LLM IT Support Chatbot for Employees.
    Completely powered by Groq LLM with conversation history.
    Accurately separates casual chat (no ticket, no troubleshooting box)
    from genuine IT issues (troubleshooting steps + ticket creation options).
    """
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message is required")

    return await groq_chat_assistant(
        message=msg,
        conversation=req.conversation,
        employee_name=req.employee_name,
    )
