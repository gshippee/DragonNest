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


@dataclass(frozen=True)
class HealthState:
    battery_pct: float = -1
    charging: bool = False
    thermal_level: float = 0
    cpu_utilization: float = 0
    accelerator_utilization: float = 0
    available_memory_mb: int = 0
    network_rtt_ms: float = 0
    reachable: bool = True
    status: HealthStatus = HealthStatus.HEALTHY


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
    steering_vector_ids: tuple[str, ...] = ()
    supported_steering_layers: tuple[int, ...] = ()
    segment: ModelSegment | None = None


@dataclass(frozen=True)
class Device:
    device_id: str
    display_name: str
    device_type: str
    platform: str
    total_memory_mb: int
    health: HealthState
    models: tuple[ModelCapability, ...]


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


@dataclass(frozen=True)
class ExecutionPlan:
    task_id: str
    execution_mode: ExecutionMode
    request_text: str
    tasks: tuple[PlannedTask, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    steering: SteeringSpec = field(default_factory=SteeringSpec)
    reducer: str = "mock_synthesis"
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
    latency_ms: int = 0

