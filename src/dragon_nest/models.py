from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    STALE = "STALE"
    OFFLINE = "OFFLINE"


class ExecutionMode(StrEnum):
    AUTO = "auto"
    SINGLE = "single"
    DATA_PARALLEL = "data_parallel"
    LAYER_PIPELINE = "layer_pipeline"


class ReducerMode(StrEnum):
    CONCAT = "concat"
    FIRST_SUCCESS = "first_success"
    MOCK_SYNTHESIS = "mock_synthesis"


class RuntimeName(StrEnum):
    MOCK = "mock"
    GENIE = "genie"
    QNN = "qnn"


class SteeringMode(StrEnum):
    """How a deployment realizes a behavior profile.

    ``supports_steering`` remains the routing flag for *dynamic activation*
    steering.  A baked or prompt profile is intentionally not equivalent to
    that flag.
    """

    RUNTIME_VECTOR = "runtime_vector"
    BAKED_PROFILE = "baked_profile"
    PROMPT_PROFILE = "prompt_profile"
    NONE = "none"


@dataclass(frozen=True)
class HealthState:
    battery_pct: float = -1
    charging: bool = False
    thermal_level: float = -1
    cpu_utilization: float = -1
    accelerator_utilization: float = -1
    gpu_utilization: float = -1
    npu_utilization: float = -1
    available_memory_mb: int = 0
    network_rtt_ms: float = -1
    reachable: bool = True
    status: HealthStatus = HealthStatus.HEALTHY


@dataclass(frozen=True)
class HardwareInventory:
    manufacturer: str = ""
    model: str = ""
    device: str = ""
    os_version: str = ""
    api_level: int = 0
    soc_manufacturer: str = ""
    soc_model: str = ""
    cpu_abis: tuple[str, ...] = ()
    cpu_core_count: int = 0
    total_storage_mb: int = 0
    available_storage_mb: int = 0
    npu_status: str = "not_probed"
    npu_name: str = ""
    qnn_runtime_version: str = ""
    compatibility_key: str = ""


@dataclass(frozen=True)
class ModelSegment:
    pipeline_id: str
    start_layer: int
    end_layer: int
    total_layers: int
    includes_embedding: bool = False
    includes_lm_head: bool = False


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    model_family: str
    role: str
    task_classes: tuple[str, ...]
    max_context_tokens: int
    warm: bool
    quality_score: float
    model_version: str = ""
    tokenizer_id: str = ""
    precision: str = ""
    boundary_format: str = ""
    steering_vector_ids: tuple[str, ...] = ()
    supported_steering_layers: tuple[int, ...] = ()
    segment: ModelSegment | None = None
    runtime_name: str = RuntimeName.MOCK.value
    runtime_version: str = ""
    supported_accelerators: tuple[str, ...] = ("cpu",)
    min_memory_mb: int = 0
    supports_steering: bool = False
    supports_data_parallel: bool = True
    supports_layer_pipeline: bool = False
    artifact_id: str = ""
    steering_modes: tuple[str, ...] = (SteeringMode.NONE.value,)
    behavior_profile_ids: tuple[str, ...] = ()
    target_compatibility_class: str = ""


@dataclass(frozen=True)
class Device:
    device_id: str
    display_name: str
    device_type: str
    platform: str
    total_memory_mb: int
    health: HealthState
    models: tuple[ModelCapability, ...]
    hardware: HardwareInventory = field(default_factory=HardwareInventory)


@dataclass(frozen=True)
class TaskProfile:
    task_class: str
    complexity: str
    privacy_tier: str
    latency_tier: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    confidence: float
    is_compound: bool
    data_parallelizable: bool
    layer_parallel_candidate: bool
    steering_requested: bool


@dataclass(frozen=True)
class SteeringSpec:
    enabled: bool = False
    vector_id: str = ""
    model_family: str = ""
    target_layer: int = 0
    alpha: float = 0.0
    positions: str = "last"
    allow_remote_vector: bool = False
    mode: str = SteeringMode.RUNTIME_VECTOR.value
    behavior_profile_id: str = ""


@dataclass(frozen=True)
class SteeringVector:
    vector_id: str
    model_family: str
    hidden_size: int
    target_layers: tuple[int, ...]
    alpha_min: float
    alpha_max: float
    positions: tuple[str, ...]
    default_layer: int
    default_alpha: float
    default_positions: str
    storage_uri: str
    safety_label: str
    allow_remote_vector: bool = False
    # Lifecycle and compatibility-boundary metadata. A vector must not be
    # assumed to transfer across model versions, quantizations, runtimes, or
    # layers; these fields make those boundaries explicit.
    model_revision: str = ""
    base_model_fingerprint: str = ""
    tokenizer_fingerprint: str = ""
    source_layer: int = -1
    extraction_method: str = ""
    positive_dataset_hash: str = ""
    negative_dataset_hash: str = ""
    normalization: str = ""
    dtype: str = ""
    checksum: str = ""
    created_at: str = ""
    creator: str = ""
    evaluation_metrics: tuple[tuple[str, float], ...] = ()
    evaluation_dataset_version: str = ""
    validated_runtimes: tuple[str, ...] = ()
    validated_quantizations: tuple[str, ...] = ()
    status: str = "draft"  # draft | calibrated | validated | deprecated | rejected


@dataclass(frozen=True)
class PlannedTask:
    shard_id: str
    request_text: str
    selected_device_id: str = ""
    selected_model_id: str = ""
    fallback_device_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    stage_index: int
    pipeline_id: str
    selected_device_id: str
    selected_model_id: str
    start_layer: int
    end_layer: int
    model_family: str = ""
    model_version: str = ""
    tokenizer_id: str = ""
    precision: str = ""
    boundary_format: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    task_id: str
    execution_mode: ExecutionMode
    request_text: str
    tasks: tuple[PlannedTask, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    steering: SteeringSpec = field(default_factory=SteeringSpec)
    origin_device_id: str = ""
    reducer: str = ReducerMode.MOCK_SYNTHESIS.value
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    execution_mode: ExecutionMode
    selected_device_id: str
    selected_model_id: str
    fallback_device_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    route_score: float


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    success: bool
    output_text: str
    device_id: str = ""
    error_code: str = ""
    error_message: str = ""
    latency_ms: int = 0
    attempt_id: str = ""
    metrics: "ExecutionMetrics | None" = None


@dataclass(frozen=True)
class ExecutionMetrics:
    model_id: str
    model_version: str
    runtime_name: str
    runtime_version: str
    accelerator: str
    execution_latency_ms: int
    error_code: str = ""
    error_message: str = ""
    observed_memory_delta_mb: int | None = None
    observed_thermal_delta: float | None = None
    artifact_load_time_ms: int | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
