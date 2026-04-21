"""
Authentication Module for Clintelligence
Email-based user registration and login
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from pydantic import BaseModel, EmailStr
import json
from pathlib import Path

# Simple file-based user storage (can be upgraded to PostgreSQL)
USERS_FILE = Path(__file__).parent / "data" / "users.json"
SESSIONS_FILE = Path(__file__).parent / "data" / "sessions.json"

# Ensure data directory exists
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============== MODELS ==============
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    organization: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    organization: Optional[str] = None
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ============== UTILITY FUNCTIONS ==============
def hash_password(password: str) -> str:
    """Hash password with salt."""
    salt = os.getenv("AUTH_SALT", "clintelligence_salt_2024")
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


def generate_token() -> str:
    """Generate secure session token."""
    return secrets.token_urlsafe(32)


def load_users() -> Dict:
    """Load users from file."""
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except:
            return {}
    return {}


def save_users(users: Dict):
    """Save users to file."""
    USERS_FILE.write_text(json.dumps(users, indent=2))


def load_sessions() -> Dict:
    """Load sessions from file."""
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except:
            return {}
    return {}


def save_sessions(sessions: Dict):
    """Save sessions to file."""
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))


# ============== AUTH FUNCTIONS ==============
def create_user(user_data: UserCreate) -> Dict:
    """Create a new user."""
    users = load_users()

    # Check if email already exists
    email_lower = user_data.email.lower()
    if email_lower in users:
        raise ValueError("Email already registered")

    # Create user
    user_id = secrets.token_hex(8)
    users[email_lower] = {
        "id": user_id,
        "email": email_lower,
        "password_hash": hash_password(user_data.password),
        "full_name": user_data.full_name,
        "organization": user_data.organization,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None
    }

    save_users(users)

    return {
        "id": user_id,
        "email": email_lower,
        "full_name": user_data.full_name,
        "organization": user_data.organization,
        "created_at": users[email_lower]["created_at"]
    }


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticate user and return user data if valid."""
    users = load_users()
    email_lower = email.lower()

    if email_lower not in users:
        return None

    user = users[email_lower]
    if user["password_hash"] != hash_password(password):
        return None

    # Update last login
    user["last_login"] = datetime.utcnow().isoformat()
    save_users(users)

    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "organization": user["organization"],
        "created_at": user["created_at"]
    }


def create_session(user_id: str, email: str) -> str:
    """Create a new session and return token."""
    sessions = load_sessions()
    token = generate_token()

    sessions[token] = {
        "user_id": user_id,
        "email": email,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }

    save_sessions(sessions)
    return token


def validate_session(token: str) -> Optional[Dict]:
    """Validate session token and return user info."""
    sessions = load_sessions()

    if token not in sessions:
        return None

    session = sessions[token]
    expires_at = datetime.fromisoformat(session["expires_at"])

    if datetime.utcnow() > expires_at:
        # Session expired, remove it
        del sessions[token]
        save_sessions(sessions)
        return None

    # Get user data
    users = load_users()
    email = session["email"]

    if email not in users:
        return None

    user = users[email]
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "organization": user["organization"]
    }


def logout_session(token: str) -> bool:
    """Remove session."""
    sessions = load_sessions()

    if token in sessions:
        del sessions[token]
        save_sessions(sessions)
        return True
    return False


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    users = load_users()
    email_lower = email.lower()

    if email_lower not in users:
        return None

    user = users[email_lower]
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "organization": user["organization"],
        "created_at": user["created_at"]
    }
