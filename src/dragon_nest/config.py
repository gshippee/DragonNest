from __future__ import annotations

from pathlib import Path

import yaml

from .models import Device, HardwareInventory, HealthState, ModelCapability, ModelSegment


def load_devices(path: str | Path) -> list[Device]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    devices: list[Device] = []
    for item in raw.get("devices", []):
        models: list[ModelCapability] = []
        for model in item.get("models", []):
            segment = ModelSegment(**model["segment"]) if model.get("segment") else None
            models.append(
                ModelCapability(
                    model_id=model["model_id"],
                    model_family=model["model_family"],
                    role=model["role"],
                    task_classes=tuple(model["task_classes"]),
                    max_context_tokens=int(model["max_context_tokens"]),
                    warm=bool(model["warm"]),
                    quality_score=float(model["quality_score"]),
                    model_version=str(model.get("model_version", "")),
                    tokenizer_id=str(model.get("tokenizer_id", "")),
                    precision=str(model.get("precision", "")),
                    boundary_format=str(model.get("boundary_format", "")),
                    steering_vector_ids=tuple(model.get("steering_vector_ids", [])),
                    supported_steering_layers=tuple(
                        int(value)
                        for value in model.get("supported_steering_layers", [])
                    ),
                    segment=segment,
                    runtime_name=str(model.get("runtime_name", "mock")),
                    runtime_version=str(model.get("runtime_version", "")),
                    supported_accelerators=tuple(
                        model.get("supported_accelerators", ["cpu"])
                    ),
                    min_memory_mb=int(model.get("min_memory_mb", 0)),
                    supports_steering=bool(
                        model.get(
                            "supports_steering",
                            bool(model.get("steering_vector_ids")),
                        )
                    ),
                    supports_data_parallel=bool(
                        model.get("supports_data_parallel", segment is None)
                    ),
                    supports_layer_pipeline=bool(
                        model.get("supports_layer_pipeline", segment is not None)
                    ),
                    artifact_id=str(model.get("artifact_id", "")),
                    steering_modes=tuple(model.get("steering_modes", ["none"])),
                    behavior_profile_ids=tuple(
                        model.get("behavior_profile_ids", [])
                    ),
                    target_compatibility_class=str(
                        model.get("target_compatibility_class", "")
                    ),
                )
            )
        devices.append(
            Device(
                device_id=item["device_id"],
                display_name=item["display_name"],
                device_type=item["device_type"],
                platform=item["platform"],
                total_memory_mb=int(item["total_memory_mb"]),
                health=HealthState(**item["health"]),
                models=tuple(models),
                hardware=HardwareInventory(**item.get("hardware", {})),
            )
        )
    return devices


def load_device(path: str | Path, device_id: str) -> Device:
    for device in load_devices(path):
        if device.device_id == device_id:
            return device
    raise KeyError(f"device {device_id} not found in {path}")
