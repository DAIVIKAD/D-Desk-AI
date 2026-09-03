"""
D Desk AI — FastAPI Application Entry Point
──────────────────────────────────────────────
Smart Helpdesk Ticketing Solution for IT Services

Run:
  python run.py

Server starts at:
  http://localhost:8000

Architecture:
  FastAPI microservice with API routers
  Advanced ML Classifier v4.0 (TF-IDF + Logistic Regression)
    → 17 category features + 13 priority features
    → Category-specific bigram dictionaries
    → Confidence calibration + Explainability
  Groq/LLaMA for LLM features
"""

import uvicorn
import os
from app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("D_DESK_RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}

    print("\n" + "═" * 62)
    print("  D Desk AI — Smart Helpdesk Ticketing Solution")
    print("  FastAPI + Cloud Firestore Architecture v4.0.0")
    print("─" * 62)
    print("  Database : Google Cloud Firestore")
    print("  ML       : Pretrained TF-IDF + Logistic Regression")
    print("  Vision   : In-memory CNN MobileNetV2 Hardware Defect Detection")
    print("  LLM      : Groq LLaMA 3.1 Cloud Intelligence")
    print("═" * 62)
    print(f"  → Server listening on 0.0.0.0:{port}")
    print(f"  → API Health: http://0.0.0.0:{port}/api/health")
    print(f"  → Swagger Docs: http://0.0.0.0:{port}/docs")
    print(f"  → Reload Mode: {'ON' if reload_enabled else 'OFF'}")
    print("═" * 62 + "\n")

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload_enabled)
