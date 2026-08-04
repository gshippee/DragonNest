from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

from dragon_nest.config import load_devices
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.transport.agent import certificate_fingerprint
from dragon_nest.transport.brain import BrainService, BrainServiceConfig


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
