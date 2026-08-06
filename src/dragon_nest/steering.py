from __future__ import annotations

from pathlib import Path

import yaml

from .models import ModelCapability, SteeringSpec, SteeringVector


class SteeringRegistry:
    def __init__(self, vectors: dict[str, SteeringVector]):
        self._vectors = vectors

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SteeringRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        vectors = {}
        for item in raw.get("vectors", []):
            vector = SteeringVector(
                vector_id=item["vector_id"],
                model_family=item["model_family"],
                hidden_size=int(item["hidden_size"]),
                target_layers=tuple(int(x) for x in item["target_layers"]),
                alpha_min=float(item["alpha_min"]),
                alpha_max=float(item["alpha_max"]),
                positions=tuple(item["positions"]),
                default_layer=int(item.get("default_layer", item["target_layers"][0])),
                default_alpha=float(item.get("default_alpha", 0.0)),
                default_positions=item.get("default_positions", "last"),
                storage_uri=item.get("storage_uri", ""),
                safety_label=item.get("safety_label", ""),
                allow_remote_vector=bool(item.get("allow_remote_vector", False)),
                model_revision=str(item.get("model_revision", "")),
                base_model_fingerprint=str(item.get("base_model_fingerprint", "")),
                tokenizer_fingerprint=str(item.get("tokenizer_fingerprint", "")),
                source_layer=int(item.get("source_layer", -1)),
                extraction_method=str(item.get("extraction_method", "")),
                positive_dataset_hash=str(item.get("positive_dataset_hash", "")),
                negative_dataset_hash=str(item.get("negative_dataset_hash", "")),
                normalization=str(item.get("normalization", "")),
                dtype=str(item.get("dtype", "")),
                checksum=str(item.get("checksum", "")),
                created_at=str(item.get("created_at", "")),
                creator=str(item.get("creator", "")),
                evaluation_metrics=tuple(
                    (str(name), float(value))
                    for name, value in (item.get("evaluation_metrics") or {}).items()
                ),
                evaluation_dataset_version=str(
                    item.get("evaluation_dataset_version", "")
                ),
                validated_runtimes=tuple(item.get("validated_runtimes", [])),
                validated_quantizations=tuple(
                    str(value) for value in item.get("validated_quantizations", [])
                ),
                status=str(item.get("status", "draft")),
            )
            vectors[vector.vector_id] = vector
        return cls(vectors)

    def default_spec(self, vector_id: str) -> SteeringSpec:
        vector = self._vectors[vector_id]
        return SteeringSpec(
            enabled=True,
            vector_id=vector.vector_id,
            model_family=vector.model_family,
            target_layer=vector.default_layer,
            alpha=vector.default_alpha,
            positions=vector.default_positions,
            allow_remote_vector=vector.allow_remote_vector,
        )

    def vectors(self) -> tuple[SteeringVector, ...]:
        return tuple(self._vectors[key] for key in sorted(self._vectors))

    _ROUTABLE_STATUSES = frozenset({"calibrated", "validated"})

    def runtime_compatible(
        self,
        vector_id: str,
        model_family: str,
        model_revision: str,
        runtime: str,
        quantization: str,
        injection_layer: int,
    ) -> tuple[bool, str]:
        """Check whether a vector may be injected on a concrete deployment.

        A vector is never assumed to transfer across model versions,
        quantizations, runtimes, or layers; only explicitly validated
        combinations pass.
        """
        vector = self._vectors.get(vector_id)
        if vector is None:
            return False, f"unknown steering vector {vector_id}"
        if vector.status not in self._ROUTABLE_STATUSES:
            return (
                False,
                f"vector {vector_id} status is {vector.status!r}; only "
                "calibrated or validated vectors are routable",
            )
        if vector.model_family != model_family:
            return (
                False,
                f"vector family {vector.model_family} does not match model "
                f"family {model_family}",
            )
        if vector.model_revision and model_revision and (
            vector.model_revision != model_revision
        ):
            return (
                False,
                f"vector was extracted from revision {vector.model_revision}, "
                f"not {model_revision}; cross-revision transfer is not assumed",
            )
        if vector.validated_runtimes and runtime not in vector.validated_runtimes:
            return (
                False,
                f"vector {vector_id} is not validated for runtime {runtime!r} "
                f"(validated: {', '.join(vector.validated_runtimes)})",
            )
        if vector.validated_quantizations and (
            quantization not in vector.validated_quantizations
        ):
            return (
                False,
                f"vector {vector_id} is not validated for quantization "
                f"{quantization!r} "
                f"(validated: {', '.join(vector.validated_quantizations)})",
            )
        if injection_layer not in vector.target_layers:
            return (
                False,
                f"vector {vector_id} does not support injection layer "
                f"{injection_layer}",
            )
        return True, f"vector {vector_id} validated for {runtime}/{quantization}"

    def validate(self, spec: SteeringSpec, model: ModelCapability) -> tuple[bool, str]:
        if not spec.enabled:
            return True, "steering disabled"
        vector = self._vectors.get(spec.vector_id)
        if vector is None:
            return False, f"unknown steering vector {spec.vector_id}"
        if vector.status in {"deprecated", "rejected"}:
            return (
                False,
                f"steering vector {spec.vector_id} status is {vector.status}",
            )
        if vector.model_family != model.model_family and model.model_family != "mock":
            return (
                False,
                f"vector family {vector.model_family} does not match model family {model.model_family}",
            )
        if spec.vector_id not in model.steering_vector_ids:
            return (
                False,
                f"model {model.model_id} does not advertise vector {spec.vector_id}",
            )
        if spec.target_layer not in vector.target_layers:
            return (
                False,
                f"vector {spec.vector_id} does not support layer {spec.target_layer}",
            )
        if spec.target_layer not in model.supported_steering_layers:
            return (
                False,
                f"model {model.model_id} does not support steering layer {spec.target_layer}",
            )
        if not (vector.alpha_min <= spec.alpha <= vector.alpha_max):
            return (
                False,
                f"alpha {spec.alpha} outside range {vector.alpha_min}..{vector.alpha_max}",
            )
        if spec.positions not in vector.positions:
            return (
                False,
                f"positions {spec.positions} not supported by vector {spec.vector_id}",
            )
        return (
            True,
            f"steering vector {spec.vector_id} compatible with {model.model_id}",
        )
