"""Role-based access control.

Users hold roles. Roles grant read and write access to departments. Unknown
roles and unknown users get nothing (fail closed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

ALL = "*"


class UnknownUserError(KeyError):
    pass


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    roles: tuple[str, ...]
    email: str | None = None


def _parse_grants(mapping: dict[str, list[str]], departments: frozenset[str]) -> dict[str, frozenset[str]]:
    grants: dict[str, frozenset[str]] = {}
    for role, depts in (mapping or {}).items():
        if ALL in depts:
            grants[role] = departments
            continue
        unknown = set(depts) - departments
        if unknown:
            raise ValueError(f"role {role!r} grants unknown departments: {sorted(unknown)}")
        grants[role] = frozenset(depts)
    return grants


class AccessPolicy:
    def __init__(self, role_grants: dict[str, list[str]], departments: Iterable[str],
                 write_grants: dict[str, list[str]] | None = None,
                 analytics_roles: Iterable[str] = ()):
        self.departments = frozenset(departments)
        self._grants = _parse_grants(role_grants, self.departments)
        self._write = _parse_grants(write_grants or {}, self.departments)
        self.analytics_roles = frozenset(analytics_roles)

    def allowed_departments(self, roles: Iterable[str]) -> frozenset[str]:
        allowed: set[str] = set()
        for role in roles:
            allowed |= self._grants.get(role, frozenset())  # unknown role -> nothing
        return frozenset(allowed)

    def writable_departments(self, roles: Iterable[str]) -> frozenset[str]:
        writable: set[str] = set()
        for role in roles:
            writable |= self._write.get(role, frozenset())
        return frozenset(writable)

    def can_read(self, user: User, department: str) -> bool:
        return department in self.allowed_departments(user.roles)

    def can_view_analytics(self, user: User) -> bool:
        return bool(self.analytics_roles.intersection(user.roles))


class UserDirectory:
    def __init__(self, users: Iterable[User]):
        self._users = {u.user_id: u for u in users}
        self._by_email = {u.email.strip().lower(): u for u in self._users.values() if u.email}

    def get(self, user_id: str) -> User:
        try:
            return self._users[user_id]
        except KeyError:
            raise UnknownUserError(user_id) from None

    def get_by_email(self, email: str) -> User:
        try:
            return self._by_email[email.strip().lower()]
        except (KeyError, AttributeError):
            raise UnknownUserError(email) from None

    def all(self) -> list[User]:
        return list(self._users.values())


def load_access_config(path: Path) -> tuple[AccessPolicy, UserDirectory]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy = AccessPolicy(
        data["role_grants"],
        data["departments"],
        write_grants=data.get("write_grants"),
        analytics_roles=data.get("analytics_roles", []),
    )
    users = [
        User(user_id=uid, name=u["name"], roles=tuple(u["roles"]), email=u.get("email"))
        for uid, u in data["users"].items()
    ]
    return policy, UserDirectory(users)
