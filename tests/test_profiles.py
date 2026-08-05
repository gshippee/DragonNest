from __future__ import annotations

import sqlite3
from pathlib import Path

from dragon_nest.profiles import ProfileStore
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.steering import SteeringRegistry
from dragon_nest.transport.brain import BrainService


ROOT = Path(__file__).resolve().parents[1]


def test_personal_profile_and_device_association_persist(tmp_path):
    database = tmp_path / "state.sqlite3"
    store = ProfileStore(database)
    profile = store.create(
        person_name="Alex",
        preferred_mode="private",
        steering_vector_id="concise-vs-verbose-layer-7",
        steering_alpha=-2.0,
        steering_positions="last",
        notes="Personal phone",
    )
    store.associate_device("phone-01", profile.profile_id, "Alex's Phone")
    store.close()

    reopened = ProfileStore(database)
    loaded = reopened.profile_for_device("phone-01")
    association = reopened.association_for_device("phone-01")

    assert loaded is not None
    assert loaded.person_name == "Alex"
    assert loaded.preferred_mode == "private"
    assert loaded.steering_vector_id == "concise-vs-verbose-layer-7"
    assert loaded.steering_alpha == -2.0
    assert association is not None
    assert association.device_name == "Alex's Phone"

    updated = reopened.update(profile.profile_id, steering_alpha=-1.25)
    assert updated.steering_alpha == -1.25
    reopened.close()


def test_profile_store_migrates_legacy_persona_column(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE personal_profiles (
            profile_id TEXT PRIMARY KEY,
            person_name TEXT NOT NULL,
            preferred_mode TEXT NOT NULL,
            steering_vector_id TEXT NOT NULL,
            steering_alpha REAL NOT NULL,
            steering_positions TEXT NOT NULL,
            allow_remote_vector INTEGER NOT NULL,
            notes TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE device_profiles (
            device_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL REFERENCES personal_profiles(profile_id),
            device_name TEXT NOT NULL,
            associated_at REAL NOT NULL
        );
        INSERT INTO personal_profiles VALUES (
            'profile-1', 'Alex', 'auto', 'concise-vs-verbose-layer-7',
            2.0, 'last', 0, 'Likes direct answers', 1.0, 1.0
        );
        """
    )
    connection.commit()
    connection.close()

    store = ProfileStore(database)

    assert store.get("profile-1").persona_id == "detailed"
    store.close()


def test_authenticated_registration_creates_and_updates_persona_profile():
    steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    service = BrainService(steering_registry=steering)
    model = pb.ModelCapability(
        model_id="android-mock-v1",
        model_family="mock",
        role="small_chat",
        task_classes=("chat_qa",),
    )
    first = pb.RegisterDevice(
        device_id="phone-01",
        display_name="Alex's Phone",
        enrollment_token="dev-token",
        models=(model,),
        personal_profile=pb.PersonalProfileRegistration(
            person_name="Alex",
            notes="Prefers practical examples",
            persona_id="concise",
        ),
    )

    assert service._registration_error(first) == ""
    created = service.profiles.profile_for_device("phone-01")
    assert created is not None
    assert created.persona_id == "concise"
    assert created.steering_alpha == -2.0

    updated = pb.RegisterDevice()
    updated.CopyFrom(first)
    updated.personal_profile.notes = "Prefers implementation details"
    updated.personal_profile.persona_id = "detailed"
    assert service._registration_error(updated) == ""

    loaded = service.profiles.profile_for_device("phone-01")
    assert loaded is not None
    assert loaded.profile_id == created.profile_id
    assert loaded.notes == "Prefers implementation details"
    assert loaded.persona_id == "detailed"
    assert loaded.steering_alpha == 2.0


def test_persona_resolution_and_profile_prompt_are_explicit():
    steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    service = BrainService(steering_registry=steering)

    concise = service._steering_for_persona("concise")
    assert concise.enabled
    assert concise.alpha == -2.0
    assert service._steering_for_persona("balanced").enabled is False
    assert service._with_profile_context("Help me plan.", "I prefer short lists.") == (
        "About the user:\nI prefer short lists.\n\nRequest:\nHelp me plan."
    )
    assert service._with_profile_context("Help me plan.", "") == "Help me plan."
