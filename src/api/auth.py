"""
This module decouples authentication (who you are) from authorization 
(what you are allowed to do). If you later switch from API keys to OAuth2/JWTs
using settings.auth_jwt_issuer, you only need to update get_current_user—every 
route using RoleChecker stays untouched.
"""

import secrets
from enum import Enum

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from src.common.config import settings


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    UNDERWRITER = "UNDERWRITER"
    AUDITOR = "AUDITOR"

class AuthenticatedUser(BaseModel):
    user_id: str
    role: UserRole

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def _get_key_registry() -> dict[str, AuthenticatedUser]:
    """Maps configured API keys to their corresponding roles."""
    return {
        settings.master_api_key.get_secret_value(): AuthenticatedUser(user_id="system-admin", role=UserRole.ADMIN),
        settings.underwriter_api_key.get_secret_value(): AuthenticatedUser(user_id="underwriter-svc", role=UserRole.UNDERWRITER),
        settings.auditor_api_key.get_secret_value(): AuthenticatedUser(user_id="compliance-auditor", role=UserRole.AUDITOR),
    }

async def get_current_user(api_key: str = Security(api_key_header)) -> AuthenticatedUser:
    """Authenticates the request and resolves the caller's role."""
    registry = _get_key_registry()
    supplied = api_key.encode("utf-8")
    matched: AuthenticatedUser | None = None
    # Compare against every key (no early exit) to keep timing independent of which key matched.
    for registered_key, user in registry.items():
        if secrets.compare_digest(supplied, registered_key.encode("utf-8")):
            matched = user
    if matched is not None:
        return matched

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

class RoleChecker:
    """Reusable FastAPI dependency that enforces role-based access."""

    def __init__(self, allowed_roles: list[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role.value}' is not authorized to access this resource.",
            )
        return current_user