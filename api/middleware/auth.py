import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer

logger = logging.getLogger("ecoloop_ai.auth")

from config.settings import settings

JWT_SECRET = settings.JWT_SECRET
JWT_ALGORITHM = "HS256"

EXEMPT_PATHS = {
    "/api/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
}

security_scheme = HTTPBearer(auto_error=False)


async def verify_token(request: Request) -> Optional[dict]:
    if request.url.path in EXEMPT_PATHS:
        logger.debug(f"Chemin exempté d'authentification : {request.url.path}")
        return None

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Token manquant")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Format invalide")

    token = parts[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM],
            options={"verify_exp": True, "verify_iat": True, "require": ["exp", "sub"]})
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")


async def get_current_user(request: Request) -> dict:
    payload = await verify_token(request)
    if payload is None:
        raise HTTPException(status_code=401, detail="Authentification requise")
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "user"),
        "payload": payload,
    }


def create_token_for_testing(
    user_id: str, email: str = "test@ecoloop.ai",
    role: str = "user", expires_in_seconds: int = 3600
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id, "email": email, "role": role,
        "iat": now, "exp": now + timedelta(seconds=expires_in_seconds),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
