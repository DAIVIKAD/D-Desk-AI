"""
D Desk AI — Firebase Admin & Cloud Firestore Service
──────────────────────────────────────────────────────
Provides clean, centralized integration with Firebase Admin SDK and Cloud Firestore.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from app.config import Config

logger = logging.getLogger("ddesk.firebase")

_firebase_app = None
_firestore_db = None
_init_error: Optional[str] = None


def _resolve_credentials_file(path_str: str) -> Optional[Path]:
    """Resolve credential file path relative to project root or absolute."""
    if not path_str:
        return None

    path = Path(path_str)
    if path.is_file():
        return path.resolve()

    # Try relative to the files/ directory (where run.py lives)
    base_dir = Path(__file__).resolve().parents[2]
    candidate = (base_dir / path_str).resolve()
    if candidate.is_file():
        return candidate

    return None


def initialize_firebase() -> bool:
    """
    Initialize Firebase Admin SDK using configured credentials.
    Supports:
      1. FIREBASE_CREDENTIALS_PATH (JSON file path)
      2. FIREBASE_CREDENTIALS_JSON (raw JSON string for cloud/Render deployment)
      3. GOOGLE_APPLICATION_CREDENTIALS (standard GCP env var)
      4. Default application credentials
    """
    global _firebase_app, _firestore_db, _init_error

    if _firebase_app is not None:
        return True

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as e:
        _init_error = f"firebase-admin package is not installed: {e}"
        logger.warning(_init_error)
        return False

    cred = None

    # Option 1: Raw JSON string in environment variable
    raw_json = Config.FIREBASE_CREDENTIALS_JSON or os.getenv("FIREBASE_CREDENTIALS_JSON", "")
    if raw_json.strip():
        try:
            cert_dict = json.loads(raw_json)
            cred = credentials.Certificate(cert_dict)
            logger.info("Firebase: Initializing with raw JSON credentials")
        except Exception as e:
            _init_error = f"Failed to parse FIREBASE_CREDENTIALS_JSON: {e}"
            logger.error(_init_error)
            return False

    # Option 2: Path to service account JSON file
    if cred is None:
        cred_path_str = Config.FIREBASE_CREDENTIALS_PATH or os.getenv("FIREBASE_CREDENTIALS_PATH", "")
        if cred_path_str:
            resolved_path = _resolve_credentials_file(cred_path_str)
            if resolved_path and resolved_path.exists():
                try:
                    cred = credentials.Certificate(str(resolved_path))
                    logger.info("Firebase: Initializing with service account file")
                except Exception as e:
                    _init_error = f"Failed to load certificate from {resolved_path.name}: {e}"
                    logger.error(_init_error)
                    return False
            else:
                _init_error = f"Credentials file not found at '{cred_path_str}'"
                logger.warning(_init_error)
                return False

    # Option 3: Standard GOOGLE_APPLICATION_CREDENTIALS or default credentials
    if cred is None:
        gcp_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if gcp_cred and Path(gcp_cred).is_file():
            try:
                cred = credentials.Certificate(gcp_cred)
                logger.info("Firebase: Initializing with GOOGLE_APPLICATION_CREDENTIALS")
            except Exception as e:
                _init_error = f"Failed to load GOOGLE_APPLICATION_CREDENTIALS: {e}"
                logger.error(_init_error)
                return False

    if cred is None:
        _init_error = "No Firebase credentials provided (FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON)"
        return False

    try:
        # Avoid re-initialization if already initialized
        try:
            _firebase_app = firebase_admin.get_app()
        except ValueError:
            _firebase_app = firebase_admin.initialize_app(cred, {
                "projectId": Config.FIREBASE_PROJECT_ID,
            })

        _firestore_db = firestore.client()
        _init_error = None
        logger.info("Firebase Admin SDK successfully connected to project: %s", Config.FIREBASE_PROJECT_ID)
        return True
    except Exception as e:
        _init_error = f"Firebase initialization failed: {e}"
        logger.error(_init_error)
        _firebase_app = None
        _firestore_db = None
        return False


def is_firebase_connected() -> bool:
    """Check if Firebase is initialized and available."""
    if _firestore_db is None:
        initialize_firebase()
    return _firestore_db is not None


def get_firestore() -> Optional[Any]:
    """Get the active Firestore client instance, or None if not configured."""
    if _firestore_db is None:
        initialize_firebase()
    return _firestore_db


def get_firebase_status() -> dict[str, Any]:
    """Return safe metadata about the database / Firebase connection status."""
    connected = is_firebase_connected()
    project_id = Config.FIREBASE_PROJECT_ID

    if connected and _firebase_app:
        try:
            project_id = _firebase_app.project_id or project_id
        except Exception:
            pass

    return {
        "connected": connected,
        "project_id": project_id if connected else None,
        "service": "Cloud Firestore",
        "storage_mode": "firestore",
        "error": _init_error if not connected else None,
    }


def test_firebase_connection() -> dict[str, Any]:
    """
    Perform a safe read/write test to verify Cloud Firestore connectivity.
    Writes a test verification document to the `_system_checks` collection
    and reads it back.
    """
    if not is_firebase_connected():
        return {
            "success": False,
            "error": _init_error or "Firebase is not connected",
            "read": False,
            "write": False,
        }

    db = get_firestore()
    doc_id = "connectivity_check"
    collection_name = "_system_checks"

    try:
        from datetime import datetime
        now_iso = datetime.utcnow().isoformat()
        test_payload = {
            "service": "D Desk AI Helpdesk",
            "check_type": "connectivity_verification",
            "timestamp": now_iso,
            "status": "healthy",
            "version": "4.0.0",
        }

        # 1. Write safe test document
        doc_ref = db.collection(collection_name).document(doc_id)
        doc_ref.set(test_payload)

        # 2. Read test document back
        doc = doc_ref.get()
        if not doc.exists:
            return {
                "success": False,
                "error": "Verification document was written but could not be read back",
                "write": True,
                "read": False,
            }

        retrieved_data = doc.to_dict()
        return {
            "success": True,
            "project_id": Config.FIREBASE_PROJECT_ID,
            "collection": collection_name,
            "document_id": doc_id,
            "written_payload": test_payload,
            "retrieved_payload": retrieved_data,
            "write": True,
            "read": True,
            "error": None,
        }
    except Exception as e:
        logger.error("Firebase connectivity test failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "write": False,
            "read": False,
        }
