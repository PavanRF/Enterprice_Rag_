"""Resolve the signed-in user from identity supplied by the SSO layer.

In production a reverse proxy (or App Service / API gateway) authenticates the
user against your identity provider and forwards their ID in a header. The
proxy MUST strip any client-supplied copy of that header, or anyone could
impersonate anyone.
"""
from __future__ import annotations

from .rbac import UnknownUserError, User, UserDirectory


# One AuthError class for every sign-in method, so the UI can catch them all the same way.
from .auth import AuthError  # noqa: E402


def user_from_header(users: UserDirectory, header_value: str | None) -> User:
    if not header_value or not header_value.strip():
        raise AuthError("no authenticated identity was provided. Sign in through the company portal.")
    try:
        return users.get(header_value.strip())
    except UnknownUserError:
        raise AuthError(f"'{header_value.strip()}' does not have access to this assistant.") from None


def user_from_google(users: UserDirectory, email: str | None) -> User:
    """Only people listed in config/access.yaml (with their work email) get in.
    A valid Google account on its own is not enough."""
    if not email or not email.strip():
        raise AuthError("Google did not provide an email address for this account.")
    try:
        return users.get_by_email(email)
    except UnknownUserError:
        raise AuthError(f"{email.strip()} is not authorized for this assistant. "
                        "Ask IT to add your account.") from None
