"""
D Desk AI — FastAPI Application Factory
──────────────────────────────────────────
Microservice architecture using FastAPI APIRouters backed by Google Cloud Firestore.

Services:
  - Cloud Firestore Database Layer (Users, Tickets, Replies, Comments, Circulars, Resets)
  - Pretrained ML Classifier (TF-IDF + Logistic Regression)
  - Pretrained CNN Image Recognition (In-memory MobileNetV2)
  - Groq LLM (LLaMA 3.1)
  - Duplicate Detection (Cosine Similarity)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.services.firebase_service import initialize_firebase
from app.services.firestore_db import seed_default_users_if_empty

logger = logging.getLogger("ddesk")

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Firebase Admin SDK (required)
    initialized = initialize_firebase()
    if not initialized:
        raise RuntimeError(
            "FATAL: Firebase Admin SDK failed to initialize. "
            "Cloud Firestore is the required database for this application. "
            "Set FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON in your environment."
        )
    logger.info("Firebase Admin SDK initialized — Cloud Firestore connected.")
    # Seed demo users from environment variables if they don't exist
    seed_default_users_if_empty()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="D Desk AI API",
        description="Smart Helpdesk Ticketing Solution — IT Services",
        version="4.0.0",
        lifespan=lifespan,
    )

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register APIRouters (Microservices) ──
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.tickets import tickets_bp
    from app.routes.chat import chat_bp
    from app.routes.insights import insights_bp
    from app.routes.circulars import circulars_bp
    from app.routes.analytics import analytics_bp
    from app.routes.health import health_bp
    from app.routes.image import image_bp
    from app.routes.ml_routes import ml_routes_bp
    from app.routes.comments import comments_bp
    from app.routes.platform_admin import platform_admin_bp

    app.include_router(auth_bp)
    app.include_router(admin_bp)
    app.include_router(tickets_bp)
    app.include_router(chat_bp)
    app.include_router(insights_bp)
    app.include_router(circulars_bp)
    app.include_router(analytics_bp)
    app.include_router(health_bp)
    app.include_router(image_bp)
    app.include_router(ml_routes_bp)
    app.include_router(comments_bp)
    app.include_router(platform_admin_bp)

    # ── Serve Frontend Static Files (Strict Allowlist) ──
    base_dir = Path(__file__).resolve().parents[1]
    allowed_html_files = {
        "index.html",
        "employee.html",
        "admin.html",
        "technician.html",
        "platform-admin.html",
    }

    @app.get("/")
    async def serve_index():
        return FileResponse(base_dir / "index.html", headers=NO_CACHE_HEADERS)

    @app.get("/{filename:path}")
    async def serve_static(filename: str):
        clean_filename = filename.strip("/\\")
        if not clean_filename or ".." in clean_filename or clean_filename.startswith("."):
            raise HTTPException(status_code=404, detail="File not found")

        # Allowed frontend pages and static assets
        if (
            clean_filename in allowed_html_files
            or clean_filename in {"favicon.ico", "favicon.png"}
            or clean_filename.startswith("assets/")
        ):
            target_path = (base_dir / clean_filename).resolve()
            if target_path.is_file() and (target_path.parent == base_dir or base_dir in target_path.parents):
                return FileResponse(target_path, headers=NO_CACHE_HEADERS)

        raise HTTPException(status_code=404, detail="File not found")

    return app


app = create_app()
