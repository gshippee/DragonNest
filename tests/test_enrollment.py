from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from dragon_nest.config import load_devices
from dragon_nest.enrollment import EnrollmentManager, EnrollmentStatus
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.transport.agent import (
    AgentClientConfig,
    DeviceAgent,
    certificate_fingerprint,
)
from dragon_nest.transport.brain import (
    BrainService,
    BrainServiceConfig,
    create_server,
    stop_server,
)


ROOT = Path(__file__).resolve().parents[1]


def _registration(fingerprint: str) -> pb.RegisterDevice:
    return pb.RegisterDevice(
        device_id="phone-01",
        certificate_fingerprint=fingerprint,
        models=[pb.ModelCapability(model_id="mock-model")],
    )


def test_production_enrollment_requires_matching_verified_fingerprint():
    fingerprint = certificate_fingerprint(b"certificate-der")
    service = BrainService(BrainServiceConfig(dev_mode=False))

    assert "verified client certificate" in service._registration_error(
        _registration(fingerprint), ""
    )
    assert "does not match" in service._registration_error(
        _registration("sha256:wrong"), fingerprint
    )
    assert service._registration_error(_registration(fingerprint), fingerprint) == ""

    service.revoked_certificate_fingerprints.add(fingerprint)
    assert "revoked" in service._registration_error(
        _registration(fingerprint), fingerprint
    )


def test_certificate_fingerprint_is_sha256_and_revocation_offlines_device():
    certificate = b"test-certificate"
    fingerprint = certificate_fingerprint(certificate)
    assert fingerprint == f"sha256:{hashlib.sha256(certificate).hexdigest()}"

    async def scenario() -> None:
        service = BrainService()
        device = replace(load_devices(ROOT / "configs/dev-fabric.yaml")[0])
        service.registry.register(device)
        service.certificate_fingerprints[device.device_id] = fingerprint

        affected = await service.revoke_certificate(fingerprint)

        assert affected == (device.device_id,)
        assert service.registry.get(device.device_id).status.value == "OFFLINE"

    asyncio.run(scenario())


def test_qr_enrollment_is_expiring_device_bound_and_reconnectable():
    now = [1_000.0]
    manager = EnrollmentManager(default_ttl_seconds=120, clock=lambda: now[0])
    session = manager.create(
        brain_host="192.168.1.20",
        brain_port=50051,
        use_tls=False,
        profile_id="profile-1",
    )
    payload = json.loads(session.qr_payload())

    assert payload["type"] == "dragonnest.enrollment"
    assert payload["brain_host"] == "192.168.1.20"
    assert payload["credential"].startswith("dn_bootstrap_")
    assert payload["profile_id"] == "profile-1"
    assert "dev-token" not in session.qr_payload()

    claim = manager.claim(payload["credential"], "phone-01")
    assert claim is not None
    assert claim.device_credential.startswith("dn_device_")
    assert manager.get(session.session_id).status == EnrollmentStatus.CLAIMED
    assert manager.valid_device_credential("phone-01", claim.device_credential)
    assert not manager.valid_device_credential("phone-02", claim.device_credential)

    repeated = manager.claim(payload["credential"], "phone-01")
    assert repeated == claim
    assert manager.claim(payload["credential"], "phone-02") is None


def test_qr_enrollment_expires_and_can_be_cancelled():
    now = [2_000.0]
    manager = EnrollmentManager(default_ttl_seconds=60, clock=lambda: now[0])
    expired = manager.create(
        brain_host="brain.local", brain_port=50051, use_tls=True
    )
    credential = json.loads(expired.qr_payload())["credential"]
    now[0] += 61

    assert manager.get(expired.session_id).status == EnrollmentStatus.EXPIRED
    assert manager.claim(credential, "phone-01") is None

    active = manager.create(
        brain_host="brain.local", brain_port=50051, use_tls=True
    )
    manager.cancel(active.session_id)
    assert manager.get(active.session_id).status == EnrollmentStatus.CANCELLED


def test_brain_exchanges_bootstrap_for_device_credential():
    service = BrainService()
    session = service.enrollment.create(
        brain_host="brain.local", brain_port=50051, use_tls=False
    )
    bootstrap = json.loads(session.qr_payload())["credential"]
    registration = pb.RegisterDevice(
        device_id="phone-01",
        enrollment_token=bootstrap,
        models=[pb.ModelCapability(model_id="mock-model")],
    )

    error, device_credential = service._authorize_registration(registration)
    assert error == ""
    assert device_credential.startswith("dn_device_")

    reconnect = pb.RegisterDevice(
        device_id="phone-01",
        enrollment_token=device_credential,
        models=[pb.ModelCapability(model_id="mock-model")],
    )
    assert service._authorize_registration(reconnect) == ("", "")

    stolen = pb.RegisterDevice(
        device_id="phone-02",
        enrollment_token=device_credential,
        models=[pb.ModelCapability(model_id="mock-model")],
    )
    assert "invalid enrollment token" in service._registration_error(stolen)


def test_qr_bootstrap_rotates_and_reconnects_over_grpc():
    async def scenario() -> None:
        service = BrainService()
        session = service.enrollment.create(
            brain_host="127.0.0.1", brain_port=50051, use_tls=False
        )
        bootstrap = json.loads(session.qr_payload())["credential"]
        server, port = await create_server(service, "127.0.0.1:0")
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
        first = DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=f"127.0.0.1:{port}",
                enrollment_token=bootstrap,
                heartbeat_interval_seconds=0.02,
                reconnect_initial_seconds=0.01,
            ),
        )
        first_task = asyncio.create_task(first.run_forever())
        second = None
        second_task = None
        try:
            await asyncio.wait_for(first.registered.wait(), timeout=3)
            issued = first._enrollment_token
            assert issued.startswith("dn_device_")
            assert service.enrollment.get(session.session_id).status == (
                EnrollmentStatus.CLAIMED
            )
            await first.stop()
            await asyncio.wait_for(first_task, timeout=3)

            second = DeviceAgent(
                device,
                AgentClientConfig(
                    brain_target=f"127.0.0.1:{port}",
                    enrollment_token=issued,
                    heartbeat_interval_seconds=0.02,
                    reconnect_initial_seconds=0.01,
                ),
            )
            second_task = asyncio.create_task(second.run_forever())
            await asyncio.wait_for(second.registered.wait(), timeout=3)
            assert service.registry.get(device.device_id).stream_connected
        finally:
            if second is not None:
                await second.stop()
            if second_task is not None:
                await asyncio.gather(second_task, return_exceptions=True)
            if not first_task.done():
                await first.stop()
                await asyncio.gather(first_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())
