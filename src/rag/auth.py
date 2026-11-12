"""Password login with salted PBKDF2 hashes, constant-time checks and lockout.

Credentials are stored in data/auth/credentials.json, which is git-ignored.
Login errors are deliberately generic, so they don't reveal which usernames exist.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Callable

from .rbac import UnknownUserError, User, UserDirectory

ITERATIONS = 310_000
_DUMMY_HASH = None  # built lazily; used to keep timing equal for unknown users


class AuthError(Exception):
    pass


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                    bytes.fromhex(salt_hex), ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)


def load_credentials(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def set_password(path: Path, user_id: str, password: str, min_length: int = 10) -> None:
    if len(password) < min_length:
        raise AuthError(f"password must be at least {min_length} characters")
    creds = load_credentials(path)
    creds[user_id] = hash_password(password)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class LoginLimiter:
    """Locks an account for a while after repeated failures. Clock is injectable for tests."""

    def __init__(self, max_failures: int = 5, lockout_seconds: int = 300,
                 clock: Callable[[], float] = time.monotonic):
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self.clock = clock
        self._failures: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    def is_locked(self, user_id: str) -> bool:
        until = self._locked_until.get(user_id)
        if until is None:
            return False
        if self.clock() >= until:
            self._locked_until.pop(user_id, None)
            self._failures.pop(user_id, None)
            return False
        return True

    def record_failure(self, user_id: str) -> None:
        self._failures[user_id] = self._failures.get(user_id, 0) + 1
        if self._failures[user_id] >= self.max_failures:
            self._locked_until[user_id] = self.clock() + self.lockout_seconds

    def record_success(self, user_id: str) -> None:
        self._failures.pop(user_id, None)
        self._locked_until.pop(user_id, None)


def authenticate(users: UserDirectory, credentials: dict[str, str], user_id: str,
                 password: str, limiter: LoginLimiter) -> User:
    global _DUMMY_HASH
    if limiter.is_locked(user_id):
        raise AuthError("Too many failed attempts. Try again in a few minutes.")
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("not-a-real-password")
    try:
        user = users.get(user_id)
        stored = credentials.get(user_id)
    except UnknownUserError:
        user, stored = None, None
    # Always run one hash comparison, so response time doesn't reveal whether a user exists.
    ok = verify_password(password, stored or _DUMMY_HASH)
    if user is None or stored is None or not ok:
        limiter.record_failure(user_id)
        raise AuthError("Invalid username or password.")
    limiter.record_success(user_id)
    return user
