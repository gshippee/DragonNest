from __future__ import annotations

from dragon_nest.profiles import ProfileStore


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
