from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


class EnrollmentError(ValueError):
    pass


class EnrollmentStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass
class EnrollmentSession:
    session_id: str
    brain_host: str
    brain_port: int
    use_tls: bool
    created_at: float
    expires_at: float
    bootstrap_credential: str
    profile_id: str = ""
    device_name: str = ""
    status: EnrollmentStatus = EnrollmentStatus.PENDING
    claimed_device_id: str = ""
    claimed_at: float | None = None
    issued_device_credential: str = ""

    def public_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "brain_host": self.brain_host,
            "brain_port": self.brain_port,
            "use_tls": self.use_tls,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "profile_id": self.profile_id,
            "device_name": self.device_name,
            "claimed_device_id": self.claimed_device_id,
            "claimed_at": self.claimed_at,
        }

    def qr_payload(self) -> str:
        return json.dumps(
            {
                "type": "dragonnest.enrollment",
                "version": 1,
                "brain_host": self.brain_host,
                "brain_port": self.brain_port,
                "use_tls": self.use_tls,
                "session_id": self.session_id,
                "profile_id": self.profile_id,
                "credential": self.bootstrap_credential,
                "expires_at_epoch": int(self.expires_at),
            },
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class EnrollmentClaim:
    session_id: str
    device_id: str
    device_credential: str
    profile_id: str = ""
    device_name: str = ""


_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,251}[A-Za-z0-9])?$")


class EnrollmentManager:
    """In-memory development enrollment with one-time device binding."""

    def __init__(
        self,
        *,
        default_ttl_seconds: int = 300,
        clock: Callable[[], float] = time.time,
    ):
        if default_ttl_seconds < 30:
            raise EnrollmentError("default enrollment TTL must be at least 30 seconds")
        self.default_ttl_seconds = default_ttl_seconds
        self._clock = clock
        self._sessions: dict[str, EnrollmentSession] = {}
        self._bootstrap_index: dict[str, str] = {}
        self._device_credentials: dict[str, str] = {}

    def create(
        self,
        *,
        brain_host: str,
        brain_port: int,
        use_tls: bool,
        ttl_seconds: int | None = None,
        profile_id: str = "",
        device_name: str = "",
    ) -> EnrollmentSession:
        host = brain_host.strip()
        if not host or not _HOST.fullmatch(host):
            raise EnrollmentError("brain_host must be a hostname or IP address")
        if not 1 <= brain_port <= 65535:
            raise EnrollmentError("brain_port must be between 1 and 65535")
        ttl = ttl_seconds or self.default_ttl_seconds
        if not 30 <= ttl <= 900:
            raise EnrollmentError("enrollment TTL must be between 30 and 900 seconds")
        now = self._clock()
        bootstrap = f"dn_bootstrap_{secrets.token_urlsafe(32)}"
        session = EnrollmentSession(
            session_id=str(uuid.uuid4()),
            brain_host=host,
            brain_port=brain_port,
            use_tls=use_tls,
            created_at=now,
            expires_at=now + ttl,
            bootstrap_credential=bootstrap,
            profile_id=profile_id,
            device_name=device_name.strip(),
        )
        self._sessions[session.session_id] = session
        self._bootstrap_index[_digest(bootstrap)] = session.session_id
        return session

    def get(self, session_id: str) -> EnrollmentSession:
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise EnrollmentError("enrollment session not found") from exc
        self._refresh(session)
        return session

    def cancel(self, session_id: str) -> EnrollmentSession:
        session = self.get(session_id)
        if session.status == EnrollmentStatus.PENDING:
            session.status = EnrollmentStatus.CANCELLED
            self._bootstrap_index.pop(_digest(session.bootstrap_credential), None)
        return session

    def claim(self, bootstrap_credential: str, device_id: str) -> EnrollmentClaim | None:
        session_id = self._bootstrap_index.get(_digest(bootstrap_credential))
        if not session_id:
            return None
        session = self.get(session_id)
        if self._clock() >= session.expires_at:
            return None
        if session.status == EnrollmentStatus.CLAIMED:
            if session.claimed_device_id != device_id:
                return None
            return EnrollmentClaim(
                session.session_id,
                device_id,
                session.issued_device_credential,
                session.profile_id,
                session.device_name,
            )
        if session.status != EnrollmentStatus.PENDING:
            return None
        credential = f"dn_device_{secrets.token_urlsafe(32)}"
        session.status = EnrollmentStatus.CLAIMED
        session.claimed_device_id = device_id
        session.claimed_at = self._clock()
        session.issued_device_credential = credential
        self._device_credentials[device_id] = _digest(credential)
        return EnrollmentClaim(
            session.session_id,
            device_id,
            credential,
            session.profile_id,
            session.device_name,
        )

    def valid_device_credential(self, device_id: str, credential: str) -> bool:
        expected = self._device_credentials.get(device_id)
        return bool(expected and hmac.compare_digest(expected, _digest(credential)))

    def _refresh(self, session: EnrollmentSession) -> None:
        if (
            session.status == EnrollmentStatus.PENDING
            and self._clock() >= session.expires_at
        ):
            session.status = EnrollmentStatus.EXPIRED
            self._bootstrap_index.pop(_digest(session.bootstrap_credential), None)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
