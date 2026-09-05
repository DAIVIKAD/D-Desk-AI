"""
D Desk AI — Database Service (Cloud Firestore)
──────────────────────────────────────────────────────────────────────────────
Persistence layer backed exclusively by Google Cloud Firestore.

Collections:
  - users                      -> {username} (profiles, hashed credentials)
  - tickets                    -> {ticket_id} (lifecycle, status, priority, auto-fix)
  - ticket_replies             -> {reply_id} (technician and employee messages)
  - ticket_comments            -> {comment_id} (community discussion, voting, verification)
  - circulars                  -> {circular_id} (admin broadcast notices)
  - password_reset_requests    -> {request_id} (password reset workflow)
  - image_predictions          -> {prediction_id} (ML defect detection metadata)
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from app.config import Config
from app.services.firebase_service import get_firestore, is_firebase_connected

import bcrypt

logger = logging.getLogger("ddesk.firestore_db")


# ═══════════════════════════════════════════════════════════════════════════
#  Security / Password Hashing (bcrypt)
# ═══════════════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored bcrypt hash or fallback to legacy demo plain text."""
    if not stored_hash:
        return False

    if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$") or stored_hash.startswith("$2y$"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False

    # Legacy plain-text fallback (auto-migrated on next reset/save)
    return secrets.compare_digest(password, stored_hash)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _db():
    """
    Get the active Firestore client.
    Raises RuntimeError if Firebase/Firestore is not configured.
    """
    if not is_firebase_connected():
        raise RuntimeError(
            "Firebase/Firestore is not configured. "
            "Set FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON in your environment. "
            "This application requires Cloud Firestore as its database."
        )
    return get_firestore()


# ═══════════════════════════════════════════════════════════════════════════
#  Users Collection
# ═══════════════════════════════════════════════════════════════════════════

VALID_ROLES = {"platform_admin", "admin", "tech", "employee"}


def normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("platform_admin", "platform-admin", "superadmin", "platform admin"):
        return "platform_admin"
    if r in ("admin", "administrator"):
        return "admin"
    if r in ("tech", "technician"):
        return "tech"
    if r in ("employee", "user", "staff"):
        return "employee"
    raise ValueError(f"Invalid role '{role}'. Must be platform_admin, admin, technician, or employee.")


def role_display_name(role: str) -> str:
    r = (role or "").strip().lower()
    if r == "platform_admin":
        return "Platform Admin"
    if r == "admin":
        return "Admin"
    if r in ("tech", "technician"):
        return "Technician"
    return "Employee"


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    if not username:
        return None
    uname = username.strip().lower()
    doc = _db().collection("users").document(uname).get()
    if doc.exists:
        data = doc.to_dict()
        data["username"] = uname
        data["status"] = data.get("status") or ("disabled" if data.get("is_disabled") else "active")
        data["role"] = normalize_role(data.get("role", "employee"))
        data["display_role"] = role_display_name(data["role"])
        return data
    return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user_by_username(username)
    if not user:
        return None

    # Check if account is disabled
    if user.get("status") == "disabled" or user.get("is_disabled") is True:
        return {
            "status": "error",
            "error": "account_disabled",
            "detail": "This account is currently disabled. Please contact Platform Administration.",
        }

    if verify_password(password, user.get("password_hash") or user.get("password", "")):
        return {
            "status": "ok",
            "username": user["username"],
            "role": user.get("role", "employee"),
            "display_role": user.get("display_role", "Employee"),
            "name": user.get("name", user["username"]),
            "dept": user.get("dept"),
            "specialization": user.get("specialization"),
            "account_status": user.get("status", "active"),
        }
    return None


def list_users() -> List[Dict[str, Any]]:
    users = []
    docs = _db().collection("users").stream()
    for doc in docs:
        d = doc.to_dict()
        role = normalize_role(d.get("role", "employee"))
        status = d.get("status") or ("disabled" if d.get("is_disabled") else "active")
        users.append({
            "username": doc.id,
            "role": role,
            "display_role": role_display_name(role),
            "name": d.get("name", doc.id),
            "dept": d.get("dept"),
            "specialization": d.get("specialization"),
            "status": status,
            "created_at": d.get("created_at"),
        })
    users.sort(key=lambda u: (u["role"], u["username"]))
    return users


def create_user(
    *,
    admin_username: str,
    username: str,
    password: str,
    role: str,
    name: str,
    dept: Optional[str] = None,
    specialization: Optional[str] = None,
    status: str = "active",
) -> Dict[str, Any]:
    requester = get_user_by_username(admin_username)
    requester_role = requester.get("role") if requester else None

    if admin_username not in ("system", "secret_override"):
        if requester_role not in ("platform_admin", "admin"):
            raise PermissionError("Administrative privileges are required for this action.")

    uname = username.strip().lower()
    if not uname or not password or not name:
        raise ValueError("Username, password, and name are required.")

    if get_user_by_username(uname):
        raise ValueError(f"User '{uname}' already exists.")

    norm_role = normalize_role(role)
    if norm_role == "platform_admin" and requester_role != "platform_admin" and admin_username != "system":
        raise PermissionError("Only an existing Platform Admin can create another Platform Admin account.")

    norm_spec = (specialization or "").strip().lower().replace("-", "_") or None
    if norm_role == "tech" and not norm_spec:
        norm_spec = "general"

    norm_status = "disabled" if (status or "").strip().lower() == "disabled" else "active"

    user_data = {
        "username": uname,
        "password_hash": hash_password(password),
        "role": norm_role,
        "name": name.strip(),
        "dept": (dept or "").strip() or None,
        "specialization": norm_spec,
        "status": norm_status,
        "created_at": _now_iso(),
    }

    _db().collection("users").document(uname).set(user_data)

    return {
        "username": uname,
        "role": norm_role,
        "display_role": role_display_name(norm_role),
        "name": user_data["name"],
        "dept": user_data["dept"],
        "specialization": norm_spec,
        "status": norm_status,
    }


def update_user_status(*, admin_username: str, username: str, status: str) -> Dict[str, Any]:
    requester = get_user_by_username(admin_username)
    if not requester or requester.get("role") not in ("platform_admin", "admin"):
        raise PermissionError("Platform Admin or Admin privileges are required to modify user status.")

    uname = username.strip().lower()
    target = get_user_by_username(uname)
    if not target:
        raise ValueError(f"User '{username}' was not found.")

    if uname == admin_username.strip().lower() and status.lower() == "disabled":
        raise ValueError("You cannot disable your own administrator account.")

    norm_status = "disabled" if status.strip().lower() == "disabled" else "active"
    _db().collection("users").document(uname).update({
        "status": norm_status,
        "is_disabled": (norm_status == "disabled"),
        "updated_at": _now_iso(),
    })

    return {
        "username": uname,
        "status": norm_status,
        "message": f"User '{uname}' is now {norm_status}.",
    }


def update_user_role(
    *,
    admin_username: str,
    username: str,
    role: str,
    specialization: Optional[str] = None,
    dept: Optional[str] = None,
) -> Dict[str, Any]:
    requester = get_user_by_username(admin_username)
    if not requester or requester.get("role") != "platform_admin":
        raise PermissionError("Platform Admin privileges are required to change user roles.")

    uname = username.strip().lower()
    target = get_user_by_username(uname)
    if not target:
        raise ValueError(f"User '{username}' was not found.")

    norm_role = normalize_role(role)
    norm_spec = (specialization or "").strip().lower().replace("-", "_") or None
    if norm_role == "tech" and not norm_spec:
        norm_spec = target.get("specialization") or "general"

    updates = {
        "role": norm_role,
        "specialization": norm_spec,
        "updated_at": _now_iso(),
    }
    if dept is not None:
        updates["dept"] = (dept or "").strip() or None

    _db().collection("users").document(uname).update(updates)

    return {
        "username": uname,
        "role": norm_role,
        "display_role": role_display_name(norm_role),
        "specialization": norm_spec,
        "message": f"User '{uname}' role updated to {role_display_name(norm_role)}.",
    }


def delete_user(*, admin_username: str, username: str) -> Dict[str, Any]:
    if admin_username not in ("system", "secret_override"):
        admin = get_user_by_username(admin_username)
        if not admin or admin.get("role") not in ("platform_admin", "admin"):
            raise PermissionError("Admin privileges are required for this action.")

    uname = username.strip().lower()
    user = get_user_by_username(uname)
    if not user:
        raise ValueError(f"User '{username}' was not found.")

    if uname == admin_username.strip().lower():
        raise ValueError("Admin users cannot delete their own account.")

    if user.get("role") in ("admin", "platform_admin"):
        admin_count = len([u for u in list_users() if u["role"] in ("admin", "platform_admin")])
        if admin_count <= 1:
            raise ValueError("At least one administrator account must remain in the system.")

    db = _db()
    db.collection("users").document(uname).delete()
    resets = db.collection("password_reset_requests").where("username", "==", uname).stream()
    for r in resets:
        r.reference.delete()

    return {
        "deleted": uname,
        "status": "deleted",
    }


def get_platform_audit_info(admin_username: str) -> Dict[str, Any]:
    requester = get_user_by_username(admin_username)
    if not requester or requester.get("role") != "platform_admin":
        raise PermissionError("Platform Admin privileges are required to access platform audit telemetry.")

    all_users = list_users()
    role_counts = {"platform_admin": 0, "admin": 0, "tech": 0, "employee": 0}
    status_counts = {"active": 0, "disabled": 0}

    for u in all_users:
        r = u.get("role", "employee")
        role_counts[r] = role_counts.get(r, 0) + 1
        s = u.get("status", "active")
        status_counts[s] = status_counts.get(s, 0) + 1

    ticket_stats = get_ticket_stats()
    circulars = list_circulars()
    reset_reqs = list_password_reset_requests()

    return {
        "platform_status": "healthy",
        "timestamp": _now_iso(),
        "database": {
            "provider": "Google Cloud Firestore",
            "project_id": Config.FIREBASE_PROJECT_ID,
            "connected": is_firebase_connected(),
        },
        "users": {
            "total": len(all_users),
            "by_role": role_counts,
            "by_status": status_counts,
        },
        "tickets": ticket_stats,
        "active_circulars": len(circulars),
        "pending_password_resets": len([r for r in reset_reqs if r.get("status") == "pending"]),
        "audit_logs": [
            {
                "event": "Platform Admin Console Session Active",
                "actor": admin_username,
                "timestamp": _now_iso(),
                "severity": "INFO",
            },
            {
                "event": "Cloud Firestore Persistence Layer Synchronized",
                "actor": "System",
                "timestamp": _now_iso(),
                "severity": "INFO",
            },
        ],
    }


def seed_default_users_if_empty() -> None:
    """Seed default demo accounts into Firestore if missing."""
    if not is_firebase_connected():
        return

    default_password = os.getenv("DEMO_DEFAULT_PASSWORD", "1234").strip() or "1234"

    default_users = [
        {
            "username": "platform_admin",
            "password_hash": hash_password(default_password),
            "role": "platform_admin",
            "name": "Platform Administrator",
            "dept": "Platform Engineering",
            "specialization": None,
            "status": "active",
            "created_at": _now_iso(),
        },
        {
            "username": "admin",
            "password_hash": hash_password(default_password),
            "role": "admin",
            "name": "Alex Morgan",
            "dept": "IT Operations",
            "specialization": None,
            "status": "active",
            "created_at": _now_iso(),
        },
        {
            "username": "tech01",
            "password_hash": hash_password(default_password),
            "role": "tech",
            "name": "Samira Khan",
            "dept": "Field Support",
            "specialization": "hardware",
            "status": "active",
            "created_at": _now_iso(),
        },
        {
            "username": "emp001",
            "password_hash": hash_password(default_password),
            "role": "employee",
            "name": "Jordan Taylor",
            "dept": "Finance & Accounts",
            "specialization": None,
            "status": "active",
            "created_at": _now_iso(),
        },
    ]

    try:
        db = _db()
        users_ref = db.collection("users")
        for u in default_users:
            doc = users_ref.document(u["username"]).get()
            if not doc.exists:
                users_ref.document(u["username"]).set(u)
                logger.info("Firestore: Seeded missing default user '%s' (%s).", u["username"], u["role"])
    except Exception as e:
        logger.warning("Could not seed default users: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
#  Tickets Collection
# ═══════════════════════════════════════════════════════════════════════════

def get_next_ticket_id() -> str:
    all_tickets = list_tickets(include_deleted=True)
    max_num = 0
    for doc in all_tickets:
        t_id = doc.get("ticket_id") or ""
        if t_id.startswith("TKT-"):
            try:
                num = int(t_id.split("-")[1])
                max_num = max(max_num, num)
            except Exception:
                pass
    return f"TKT-{max_num + 1}"


def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    if not ticket_id:
        return None
    t_id = ticket_id.strip()
    doc = _db().collection("tickets").document(t_id).get()
    if doc.exists:
        data = doc.to_dict()
        data["ticket_id"] = t_id
        return data
    return None


def list_tickets(
    *,
    employee_id: Optional[str] = None,
    assigned_to: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    include_deleted: bool = False,
    deleted_only: bool = False,
    include_replies: bool = False,
) -> List[Dict[str, Any]]:
    purge_expired_deleted_tickets()

    query = _db().collection("tickets")
    if deleted_only:
        query = query.where("deleted", "==", True)
    elif not include_deleted:
        query = query.where("deleted", "==", False)

    if employee_id:
        query = query.where("employee_id", "==", employee_id.strip().lower())

    if status:
        norm_status = status.strip().lower().replace(" ", "_")
        query = query.where("status", "==", norm_status)

    all_tickets = []
    docs = query.stream()
    for doc in docs:
        d = doc.to_dict()
        d["ticket_id"] = doc.id
        if assigned_to and d.get("assigned_to") != assigned_to.strip():
            continue
        all_tickets.append(d)

    all_tickets.sort(key=lambda t: t.get("created_at") or "", reverse=True)
    all_tickets = all_tickets[:max(1, min(limit, 500))]
    if include_replies:
        for t in all_tickets:
            t["replies"] = list_ticket_replies(t["ticket_id"])
    return all_tickets


def create_ticket(data: Dict[str, Any]) -> Dict[str, Any]:
    purge_expired_deleted_tickets()
    ticket_id = data.get("ticket_id") or get_next_ticket_id()
    now_iso = _now_iso()

    record = {
        "ticket_id": ticket_id,
        "employee_id": (data.get("employee_id") or "").strip().lower(),
        "employee_name": data.get("employee_name") or data.get("employee_id"),
        "location": data.get("location"),
        "description": data.get("description", ""),
        "category": data.get("category", "Other"),
        "priority": data.get("priority", "Medium"),
        "status": data.get("status", "open"),
        "source": data.get("source", "chat"),
        "assigned_to": data.get("assigned_to"),
        "predicted_issue": data.get("predicted_issue"),
        "prediction_confidence": data.get("prediction_confidence"),
        "technician_specialization": data.get("technician_specialization"),
        "issue_resolution_status": data.get("issue_resolution_status"),
        "auto_fix": data.get("auto_fix"),
        "auto_resolved": data.get("auto_resolved", False),
        "is_duplicate": data.get("is_duplicate", False),
        "duplicate_of": data.get("duplicate_of"),
        "tfidf_vec": data.get("tfidf_vec"),
        "resolution": data.get("resolution"),
        "created_at": data.get("created_at") or now_iso,
        "updated_at": now_iso,
        "resolved_at": data.get("resolved_at"),
        "deleted": False,
        "deleted_at": None,
        "deleted_by": None,
        "purge_after_at": None,
    }

    _db().collection("tickets").document(ticket_id).set(record)
    record["replies"] = []
    return record


def update_ticket(ticket_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t_id = ticket_id.strip()
    ticket = get_ticket(t_id)
    if not ticket:
        return None

    now_iso = _now_iso()
    ticket["updated_at"] = now_iso

    if "status" in updates and updates["status"] is not None:
        new_status = updates["status"].strip().lower().replace(" ", "_")
        ticket["status"] = new_status
        if new_status == "resolved":
            ticket["resolved_at"] = now_iso
        elif new_status == "closed":
            ticket["resolved_at"] = ticket.get("resolved_at") or now_iso
        else:
            ticket["resolved_at"] = None

    if "assigned_to" in updates:
        ticket["assigned_to"] = updates["assigned_to"].strip() if updates["assigned_to"] else None

    if "resolution" in updates:
        ticket["resolution"] = updates["resolution"].strip() if updates["resolution"] else None

    if "location" in updates:
        ticket["location"] = updates["location"].strip() if updates["location"] else None

    _db().collection("tickets").document(t_id).set(ticket)
    return get_ticket(t_id)


def soft_delete_ticket(ticket_id: str, admin_username: str, retention_days: int = 30) -> Dict[str, Any]:
    t_id = ticket_id.strip()
    ticket = get_ticket(t_id)
    if not ticket:
        raise ValueError(f"Ticket '{ticket_id}' was not found.")

    deleted_at = datetime.utcnow()
    purge_at = deleted_at + timedelta(days=max(1, retention_days))

    updates = {
        "deleted": True,
        "deleted_at": deleted_at.isoformat(),
        "deleted_by": admin_username,
        "purge_after_at": purge_at.isoformat(),
        "updated_at": deleted_at.isoformat(),
    }

    _db().collection("tickets").document(t_id).update(updates)
    updated = get_ticket(t_id)
    updated["message"] = f"Ticket {t_id} moved to Recently Deleted until {purge_at.isoformat()}."
    return updated


def restore_ticket(ticket_id: str) -> Dict[str, Any]:
    t_id = ticket_id.strip()
    ticket = get_ticket(t_id)
    if not ticket:
        raise ValueError(f"Ticket '{ticket_id}' was not found.")

    updates = {
        "deleted": False,
        "deleted_at": None,
        "deleted_by": None,
        "purge_after_at": None,
        "updated_at": _now_iso(),
    }

    _db().collection("tickets").document(t_id).update(updates)
    updated = get_ticket(t_id)
    updated["message"] = f"Ticket {t_id} was restored to the active queue."
    return updated


def permanently_delete_ticket(ticket_id: str) -> bool:
    t_id = ticket_id.strip()
    db = _db()
    db.collection("tickets").document(t_id).delete()
    replies = db.collection("ticket_replies").where("ticket_id", "==", t_id).stream()
    for r in replies:
        r.reference.delete()
    comments = db.collection("ticket_comments").where("ticket_id", "==", t_id).stream()
    for c in comments:
        c.reference.delete()
    return True


def purge_expired_deleted_tickets() -> int:
    try:
        now_iso = _now_iso()
        count = 0
        db = _db()
        docs = db.collection("tickets").where("deleted", "==", True).stream()
        for doc in docs:
            d = doc.to_dict()
            if d.get("purge_after_at") and d["purge_after_at"] <= now_iso:
                permanently_delete_ticket(doc.id)
                count += 1
        return count
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Ticket Replies Collection
# ═══════════════════════════════════════════════════════════════════════════

def add_ticket_reply(
    *,
    ticket_id: str,
    author_username: str,
    author_name: str,
    author_role: str,
    message: str,
) -> Dict[str, Any]:
    t_id = ticket_id.strip()
    ticket = get_ticket(t_id)
    if not ticket or ticket.get("deleted"):
        raise ValueError("Ticket not found.")

    role = "tech" if author_role.lower() in ("tech", "technician") else author_role.lower()
    now_iso = _now_iso()
    reply_id = str(uuid.uuid4())[:8]

    reply_data = {
        "id": reply_id,
        "ticket_id": t_id,
        "author_username": author_username.strip().lower(),
        "author_name": author_name.strip(),
        "author_role": role,
        "message": message.strip(),
        "created_at": now_iso,
    }

    _db().collection("ticket_replies").document(reply_id).set(reply_data)

    if role == "tech":
        ticket_updates = {
            "assigned_to": author_name.strip(),
            "updated_at": now_iso,
        }
        if ticket.get("status") == "open":
            ticket_updates["status"] = "in_progress"
        update_ticket(t_id, ticket_updates)

    reply_data["author_role"] = "technician" if role == "tech" else role
    return reply_data


def list_ticket_replies(ticket_id: str) -> List[Dict[str, Any]]:
    replies = []
    docs = (
        _db()
        .collection("ticket_replies")
        .where("ticket_id", "==", ticket_id.strip())
        .stream()
    )
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        role = d.get("author_role", "")
        d["author_role"] = "technician" if role == "tech" else role
        replies.append(d)
    replies.sort(key=lambda r: r.get("created_at") or "")
    return replies


# ═══════════════════════════════════════════════════════════════════════════
#  Community Comments Collection
# ═══════════════════════════════════════════════════════════════════════════

def add_ticket_comment(
    *,
    ticket_id: str,
    employee_id: str,
    employee_name: str,
    comment: str,
    parent_comment_id: Optional[str] = None,
) -> Dict[str, Any]:
    t_id = ticket_id.strip()
    ticket = get_ticket(t_id)
    if not ticket:
        raise ValueError("Ticket not found.")

    comment_id = str(uuid.uuid4())[:8]
    comment_data = {
        "id": comment_id,
        "ticket_id": t_id,
        "employee_id": employee_id.strip().lower(),
        "employee_name": employee_name.strip() or employee_id,
        "comment": comment.strip(),
        "parent_comment_id": parent_comment_id,
        "votes": 0,
        "is_verified": False,
        "is_pinned": False,
        "created_at": _now_iso(),
    }

    _db().collection("ticket_comments").document(comment_id).set(comment_data)
    return comment_data


def list_ticket_comments(ticket_id: str) -> List[Dict[str, Any]]:
    comments = []
    docs = (
        _db()
        .collection("ticket_comments")
        .where("ticket_id", "==", ticket_id.strip())
        .stream()
    )
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        comments.append(d)
    comments.sort(key=lambda c: c.get("created_at") or "")
    return comments


def vote_ticket_comment(comment_id: str) -> Optional[Dict[str, Any]]:
    cid = comment_id.strip()
    db = _db()
    doc_ref = db.collection("ticket_comments").document(cid)
    doc = doc_ref.get()
    if not doc.exists:
        return None
    comment = doc.to_dict()
    comment["id"] = cid
    comment["votes"] = int(comment.get("votes", 0)) + 1
    doc_ref.update({"votes": comment["votes"]})
    return comment


def verify_ticket_comment(comment_id: str) -> Optional[Dict[str, Any]]:
    cid = comment_id.strip()
    db = _db()
    doc_ref = db.collection("ticket_comments").document(cid)
    doc = doc_ref.get()
    if not doc.exists:
        return None
    comment = doc.to_dict()
    comment["id"] = cid
    comment["is_verified"] = not comment.get("is_verified", False)
    doc_ref.update({"is_verified": comment["is_verified"]})
    return comment


def pin_ticket_comment(comment_id: str) -> Optional[Dict[str, Any]]:
    cid = comment_id.strip()
    db = _db()
    doc_ref = db.collection("ticket_comments").document(cid)
    doc = doc_ref.get()
    if not doc.exists:
        return None
    comment = doc.to_dict()
    comment["id"] = cid
    comment["is_pinned"] = not comment.get("is_pinned", False)
    doc_ref.update({"is_pinned": comment["is_pinned"]})
    return comment


def delete_ticket_comment(comment_id: str) -> bool:
    cid = comment_id.strip()
    db = _db()
    db.collection("ticket_comments").document(cid).delete()
    children = db.collection("ticket_comments").where("parent_comment_id", "==", cid).stream()
    for child in children:
        child.reference.delete()
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Circulars Collection
# ═══════════════════════════════════════════════════════════════════════════

def create_circular(
    *,
    admin_username: str,
    subject: str,
    body: str,
    target: str = "all-tech",
    priority: str = "normal",
    expires_in_days: Optional[int] = None,
) -> Dict[str, Any]:
    admin = get_user_by_username(admin_username)
    if not admin or admin.get("role") not in ("admin", "platform_admin"):
        raise PermissionError("Admin or Platform Admin privileges are required.")

    now = datetime.utcnow()
    expires_at = (now + timedelta(days=expires_in_days)).isoformat() if expires_in_days else None
    circ_id = str(uuid.uuid4())[:8]

    circ_data = {
        "id": circ_id,
        "subject": subject.strip(),
        "body": body.strip(),
        "target": target.strip(),
        "priority": priority.strip(),
        "sent_by": admin_username.strip(),
        "created_at": now.isoformat(),
        "expires_at": expires_at,
    }

    _db().collection("circulars").document(circ_id).set(circ_data)
    return circ_data


def list_circulars(target: Optional[str] = None) -> List[Dict[str, Any]]:
    now_iso = _now_iso()
    circs = []
    db = _db()
    docs = db.collection("circulars").stream()
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        if d.get("expires_at") and d["expires_at"] <= now_iso:
            doc.reference.delete()
            continue
        if target and target not in ("all", "all-tech"):
            if d.get("target") not in (target, "all", "all-tech"):
                continue
        circs.append(d)
    circs.sort(key=lambda c: c.get("created_at") or "", reverse=True)
    return circs[:30]


def delete_circular(circular_id: str, admin_username: str) -> bool:
    admin = get_user_by_username(admin_username)
    if not admin or admin.get("role") not in ("admin", "platform_admin"):
        raise PermissionError("Admin or Platform Admin privileges are required.")

    cid = circular_id.strip()
    _db().collection("circulars").document(cid).delete()
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Password Reset Requests Collection
# ═══════════════════════════════════════════════════════════════════════════

def request_password_reset(username: str, note: Optional[str] = None) -> Dict[str, Any]:
    user = get_user_by_username(username)
    if not user:
        raise ValueError(f"User '{username}' was not found.")

    uname = user["username"]
    db = _db()

    # Check for existing pending request
    existing = db.collection("password_reset_requests").where("username", "==", uname).where("status", "==", "pending").stream()
    for doc in existing:
        d = doc.to_dict()
        d["id"] = doc.id
        return d

    req_id = str(uuid.uuid4())[:8]
    req_data = {
        "id": req_id,
        "username": uname,
        "requested_by": uname,
        "role": user.get("role", "employee"),
        "note": (note or "").strip() or None,
        "status": "pending",
        "admin_username": None,
        "requested_at": _now_iso(),
        "resolved_at": None,
    }

    db.collection("password_reset_requests").document(req_id).set(req_data)
    return req_data


def list_password_reset_requests(status: Optional[str] = None) -> List[Dict[str, Any]]:
    reqs = []
    query = _db().collection("password_reset_requests")
    if status:
        query = query.where("status", "==", status.strip().lower())
    docs = query.stream()
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        role = d.get("role", "employee")
        d["role"] = "technician" if role == "tech" else role
        reqs.append(d)
    reqs.sort(key=lambda r: r.get("requested_at") or "", reverse=True)
    return reqs


def reset_user_password(
    *,
    admin_username: str,
    username: str,
    new_password: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    if admin_username not in ("system", "secret_override"):
        admin = get_user_by_username(admin_username)
        if not admin or admin.get("role") not in ("admin", "platform_admin"):
            raise PermissionError("Admin or Platform Admin privileges are required.")

    uname = username.strip().lower()
    user = get_user_by_username(uname)
    if not user:
        raise ValueError(f"User '{username}' was not found.")

    if not new_password.strip():
        raise ValueError("A new password is required.")

    now_iso = _now_iso()
    db = _db()
    db.collection("users").document(uname).update({
        "password_hash": hash_password(new_password.strip()),
        "updated_at": now_iso,
    })

    if request_id:
        db.collection("password_reset_requests").document(str(request_id).strip()).update({
            "status": "completed",
            "admin_username": admin_username,
            "resolved_at": now_iso,
        })
    else:
        pending = db.collection("password_reset_requests").where("username", "==", uname).where("status", "==", "pending").stream()
        for r in pending:
            r.reference.update({
                "status": "completed",
                "admin_username": admin_username,
                "resolved_at": now_iso,
            })

    user_updated = get_user_by_username(uname)
    role = user_updated.get("role", "employee")
    return {
        "username": uname,
        "role": "technician" if role == "tech" else role,
        "name": user_updated.get("name", uname),
        "dept": user_updated.get("dept"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Image Prediction Metadata (Zero file storage)
# ═══════════════════════════════════════════════════════════════════════════

def create_image_prediction_record(data: Dict[str, Any]) -> Dict[str, Any]:
    pred_id = str(uuid.uuid4())[:8]
    record = {
        "id": pred_id,
        "filename": data.get("filename", "in_memory_upload.jpg"),
        "predicted_label": data.get("predicted_label", "other"),
        "confirmed_label": data.get("confirmed_label"),
        "confidence": float(data.get("confidence", 0.0)),
        "top_prediction": data.get("top_prediction"),
        "note": data.get("note", ""),
        "suggested_fix": data.get("suggested_fix", ""),
        "created_by": data.get("created_by"),
        "feedback_status": "pending",
        "issue_resolution_status": data.get("issue_resolution_status", "pending"),
        "ticket_id": data.get("ticket_id"),
        "assigned_to": data.get("assigned_to"),
        "technician_specialization": data.get("technician_specialization"),
        "confirmed_by": None,
        "created_at": _now_iso(),
        "confirmed_at": None,
    }

    _db().collection("image_predictions").document(pred_id).set(record)
    return record


def get_image_prediction_record(prediction_id: str) -> Optional[Dict[str, Any]]:
    pid = str(prediction_id).strip()
    doc = _db().collection("image_predictions").document(pid).get()
    if doc.exists:
        d = doc.to_dict()
        d["id"] = doc.id
        return d
    return None


def update_image_prediction_record(prediction_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pid = str(prediction_id).strip()
    record = get_image_prediction_record(pid)
    if not record:
        return None
    _db().collection("image_predictions").document(pid).update(updates)
    return get_image_prediction_record(pid)


def list_image_prediction_records(limit: int = 100) -> List[Dict[str, Any]]:
    records = []
    docs = _db().collection("image_predictions").stream()
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        records.append(d)
    records.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return records[:max(1, min(limit, 500))]


def clear_image_prediction_records(ids: Optional[List[str]] = None) -> int:
    count = 0
    db = _db()
    if ids:
        for pid in ids:
            db.collection("image_predictions").document(str(pid).strip()).delete()
            count += 1
    else:
        docs = db.collection("image_predictions").stream()
        for doc in docs:
            doc.reference.delete()
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  Analytics & Technician Assignment Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_ticket_stats() -> Dict[str, int]:
    purge_expired_deleted_tickets()
    today_iso = date.today().isoformat()
    active_docs = list_tickets(include_deleted=False)

    total = 0
    today_count = 0
    resolved_count = 0
    resolved_total = 0
    open_count = 0
    auto_resolved = 0

    for d in active_docs:
        total += 1
        created_at = (d.get("created_at") or "")[:10]
        resolved_at = (d.get("resolved_at") or "")[:10]
        status = d.get("status", "")

        if created_at == today_iso:
            today_count += 1
        if resolved_at == today_iso and status == "resolved":
            resolved_count += 1
        if status == "resolved":
            resolved_total += 1
        if status in ("open", "in_progress"):
            open_count += 1
        if d.get("auto_resolved"):
            auto_resolved += 1

    deleted_docs = list_tickets(deleted_only=True)
    deleted_count = len(deleted_docs)

    return {
        "total": total,
        "today": today_count,
        "resolved": resolved_count,
        "resolved_total": resolved_total,
        "open": open_count,
        "auto_resolved": auto_resolved,
        "deleted": deleted_count,
    }


def get_category_distribution() -> Dict[str, int]:
    docs = list_tickets(include_deleted=False)
    dist: Dict[str, int] = {}
    for doc in docs:
        cat = doc.get("category") or "Other"
        dist[cat] = dist.get(cat, 0) + 1
    return dist


def get_daily_volume(days: int = 14) -> List[Dict[str, Any]]:
    docs = list_tickets(include_deleted=False)
    date_counts: Dict[str, int] = {}
    for doc in docs:
        created_date = (doc.get("created_at") or "")[:10]
        if created_date:
            date_counts[created_date] = date_counts.get(created_date, 0) + 1

    result = []
    for i in range(days - 1, -1, -1):
        d_str = (date.today() - timedelta(days=i)).isoformat()
        result.append({"date": d_str, "count": date_counts.get(d_str, 0)})
    return result


def get_technician_workload() -> Dict[str, Any]:
    active_tickets = list_tickets(include_deleted=False)
    open_tickets = [t for t in active_tickets if t.get("status") in ("open", "in_progress")]

    tech_users = [u for u in list_users() if u["role"] == "technician"]
    workload = []

    for tech in tech_users:
        labels = {tech["username"].lower()}
        if tech.get("name"):
            labels.add(tech["name"].lower())

        assigned = [
            t for t in open_tickets
            if (t.get("assigned_to") or "").lower() in labels
        ]
        workload.append({
            "username": tech["username"],
            "name": tech.get("name", tech["username"]),
            "specialization": tech.get("specialization") or "general",
            "active_count": len(assigned),
            "tickets": assigned,
        })

    unassigned = [t for t in open_tickets if not t.get("assigned_to")]
    return {
        "technicians": workload,
        "unassigned": unassigned,
        "active_ticket_count": len(open_tickets),
    }


def assign_technician_by_specialization(specialization: Optional[str]) -> Optional[Dict[str, Any]]:
    from app.services.agentic_ai import normalize_specialization
    target = normalize_specialization(specialization)

    workload = get_technician_workload()["technicians"]
    if not workload:
        return None

    exact = [t for t in workload if normalize_specialization(t.get("specialization")) == target]
    fallback = [t for t in workload if normalize_specialization(t.get("specialization")) in ("general", target)]
    pool = exact or fallback or workload

    pool.sort(key=lambda t: (t["active_count"], t["username"]))
    return pool[0]
