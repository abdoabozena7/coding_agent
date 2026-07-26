"""Loopback-only authentication and request validation."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionSecurity:
    session_id: str
    lifetime_seconds: int = 12 * 60 * 60
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    cookie_value: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    csrf_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    created_at: float = field(default_factory=time.monotonic)
    active: bool = True

    @property
    def expired(self) -> bool:
        return not self.active or time.monotonic() - self.created_at > self.lifetime_seconds

    def validate_handshake(self, token: str) -> bool:
        return not self.expired and secrets.compare_digest(str(token), self.token)

    def validate_cookie(self, cookie: str | None) -> bool:
        return (
            bool(cookie)
            and not self.expired
            and secrets.compare_digest(str(cookie), self.cookie_value)
        )

    def validate_csrf(self, cookie: str | None, header: str | None) -> bool:
        return (
            bool(cookie)
            and bool(header)
            and not self.expired
            and secrets.compare_digest(str(cookie), self.csrf_token)
            and secrets.compare_digest(str(header), self.csrf_token)
        )

    def invalidate(self) -> None:
        self.active = False
        self.token = ""
        self.cookie_value = ""
        self.csrf_token = ""
