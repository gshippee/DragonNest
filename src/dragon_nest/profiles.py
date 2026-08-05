from __future__ import annotations

import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class PersonalProfile:
    profile_id: str
    person_name: str
    preferred_mode: str
    steering_vector_id: str
    steering_alpha: float
    steering_positions: str
    allow_remote_vector: bool
    notes: str
    persona_id: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    profile_id: str
    device_name: str
    associated_at: float


class ProfileStore:
    """SQLite-backed personal profiles and enrolled-device associations."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS personal_profiles (
                    profile_id TEXT PRIMARY KEY,
                    person_name TEXT NOT NULL,
                    preferred_mode TEXT NOT NULL,
                    steering_vector_id TEXT NOT NULL,
                    steering_alpha REAL NOT NULL,
                    steering_positions TEXT NOT NULL,
                    allow_remote_vector INTEGER NOT NULL,
                    notes TEXT NOT NULL,
                    persona_id TEXT NOT NULL DEFAULT 'balanced',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS device_profiles (
                    device_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL REFERENCES personal_profiles(profile_id),
                    device_name TEXT NOT NULL,
                    associated_at REAL NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(personal_profiles)"
                ).fetchall()
            }
            if "persona_id" not in columns:
                self._connection.execute(
                    "ALTER TABLE personal_profiles "
                    "ADD COLUMN persona_id TEXT NOT NULL DEFAULT 'balanced'"
                )
                self._connection.execute(
                    """
                    UPDATE personal_profiles
                    SET persona_id = CASE
                        WHEN steering_vector_id = '' THEN 'balanced'
                        WHEN steering_alpha < 0 THEN 'concise'
                        WHEN steering_alpha > 0 THEN 'detailed'
                        ELSE 'balanced'
                    END
                    """
                )

    def create(
        self,
        *,
        person_name: str,
        preferred_mode: str = "auto",
        steering_vector_id: str = "",
        steering_alpha: float = 0.0,
        steering_positions: str = "last",
        allow_remote_vector: bool = False,
        notes: str = "",
        persona_id: str = "",
    ) -> PersonalProfile:
        values = _validated_values(
            person_name=person_name,
            preferred_mode=preferred_mode,
            steering_vector_id=steering_vector_id,
            steering_alpha=steering_alpha,
            steering_positions=steering_positions,
            allow_remote_vector=allow_remote_vector,
            notes=notes,
            persona_id=persona_id,
        )
        now = time.time()
        profile_id = str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO personal_profiles (
                    profile_id, person_name, preferred_mode, steering_vector_id,
                    steering_alpha, steering_positions, allow_remote_vector,
                    notes, persona_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    *values,
                    now,
                    now,
                ),
            )
        return self.get(profile_id)

    def update(self, profile_id: str, **changes) -> PersonalProfile:
        current = self.get(profile_id)
        values = _validated_values(
            person_name=changes.get("person_name", current.person_name),
            preferred_mode=changes.get("preferred_mode", current.preferred_mode),
            steering_vector_id=changes.get(
                "steering_vector_id", current.steering_vector_id
            ),
            steering_alpha=changes.get("steering_alpha", current.steering_alpha),
            steering_positions=changes.get(
                "steering_positions", current.steering_positions
            ),
            allow_remote_vector=changes.get(
                "allow_remote_vector", current.allow_remote_vector
            ),
            notes=changes.get("notes", current.notes),
            persona_id=changes.get("persona_id", current.persona_id),
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE personal_profiles
                SET person_name=?, preferred_mode=?, steering_vector_id=?,
                    steering_alpha=?, steering_positions=?, allow_remote_vector=?,
                    notes=?, persona_id=?, updated_at=?
                WHERE profile_id=?
                """,
                (*values, time.time(), profile_id),
            )
        return self.get(profile_id)

    def get(self, profile_id: str) -> PersonalProfile:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM personal_profiles WHERE profile_id=?", (profile_id,)
            ).fetchone()
        if row is None:
            raise ProfileError("personal profile not found")
        return _profile(row)

    def all(self) -> tuple[PersonalProfile, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM personal_profiles ORDER BY created_at, profile_id"
            ).fetchall()
        return tuple(_profile(row) for row in rows)

    def associate_device(
        self, device_id: str, profile_id: str, device_name: str
    ) -> DeviceProfile:
        self.get(profile_id)
        name = device_name.strip()
        if not device_id.strip() or not name:
            raise ProfileError("device_id and device_name are required")
        associated_at = time.time()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO device_profiles VALUES (?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    profile_id=excluded.profile_id,
                    device_name=excluded.device_name,
                    associated_at=excluded.associated_at
                """,
                (device_id, profile_id, name, associated_at),
            )
        return DeviceProfile(device_id, profile_id, name, associated_at)

    def association_for_device(self, device_id: str) -> DeviceProfile | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM device_profiles WHERE device_id=?", (device_id,)
            ).fetchone()
        if row is None:
            return None
        return DeviceProfile(
            device_id=row["device_id"],
            profile_id=row["profile_id"],
            device_name=row["device_name"],
            associated_at=row["associated_at"],
        )

    def profile_for_device(self, device_id: str) -> PersonalProfile | None:
        association = self.association_for_device(device_id)
        return self.get(association.profile_id) if association else None

    def delete_if_unassociated(self, profile_id: str) -> bool:
        with self._lock, self._connection:
            associated = self._connection.execute(
                "SELECT 1 FROM device_profiles WHERE profile_id=? LIMIT 1",
                (profile_id,),
            ).fetchone()
            if associated is not None:
                return False
            deleted = self._connection.execute(
                "DELETE FROM personal_profiles WHERE profile_id=?", (profile_id,)
            )
        return deleted.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _validated_values(
    *,
    person_name: str,
    preferred_mode: str,
    steering_vector_id: str,
    steering_alpha: float,
    steering_positions: str,
    allow_remote_vector: bool,
    notes: str,
    persona_id: str,
) -> tuple[str, str, str, float, str, int, str, str]:
    name = person_name.strip()
    mode = preferred_mode.strip()
    positions = steering_positions.strip()
    if not name or len(name) > 120:
        raise ProfileError("person_name must contain 1 to 120 characters")
    if mode not in {"auto", "fast", "private", "quality", "parallel"}:
        raise ProfileError("preferred_mode is invalid")
    if not math.isfinite(steering_alpha):
        raise ProfileError("steering_alpha must be finite")
    if positions not in {"last", "all"}:
        raise ProfileError("steering_positions is invalid")
    persona = persona_id.strip()
    if not persona:
        persona = (
            "balanced"
            if not steering_vector_id.strip() or steering_alpha == 0
            else ("concise" if steering_alpha < 0 else "detailed")
        )
    if persona not in {"balanced", "concise", "detailed"}:
        raise ProfileError("persona_id is invalid")
    return (
        name,
        mode,
        steering_vector_id.strip(),
        float(steering_alpha),
        positions,
        int(allow_remote_vector),
        notes.strip()[:500],
        persona,
    )


def _profile(row: sqlite3.Row) -> PersonalProfile:
    return PersonalProfile(
        profile_id=row["profile_id"],
        person_name=row["person_name"],
        preferred_mode=row["preferred_mode"],
        steering_vector_id=row["steering_vector_id"],
        steering_alpha=row["steering_alpha"],
        steering_positions=row["steering_positions"],
        allow_remote_vector=bool(row["allow_remote_vector"]),
        notes=row["notes"],
        persona_id=row["persona_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
