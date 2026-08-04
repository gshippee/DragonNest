from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DeviceToBrain(_message.Message):
    __slots__ = ("register_device", "health_update", "task_result", "shutdown", "partial_task_result", "pipeline_stage_result")
    REGISTER_DEVICE_FIELD_NUMBER: _ClassVar[int]
    HEALTH_UPDATE_FIELD_NUMBER: _ClassVar[int]
    TASK_RESULT_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_TASK_RESULT_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_STAGE_RESULT_FIELD_NUMBER: _ClassVar[int]
    register_device: RegisterDevice
    health_update: HealthUpdate
    task_result: TaskResult
    shutdown: ShutdownEvent
    partial_task_result: PartialTaskResult
    pipeline_stage_result: PipelineStageResult
    def __init__(self, register_device: _Optional[_Union[RegisterDevice, _Mapping]] = ..., health_update: _Optional[_Union[HealthUpdate, _Mapping]] = ..., task_result: _Optional[_Union[TaskResult, _Mapping]] = ..., shutdown: _Optional[_Union[ShutdownEvent, _Mapping]] = ..., partial_task_result: _Optional[_Union[PartialTaskResult, _Mapping]] = ..., pipeline_stage_result: _Optional[_Union[PipelineStageResult, _Mapping]] = ...) -> None: ...

class BrainToDevice(_message.Message):
    __slots__ = ("registration_accepted", "registration_rejected", "execute_task", "cancel_task", "heartbeat_ack", "execute_shard", "execute_pipeline_stage")
    REGISTRATION_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_REJECTED_FIELD_NUMBER: _ClassVar[int]
    EXECUTE_TASK_FIELD_NUMBER: _ClassVar[int]
    CANCEL_TASK_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_ACK_FIELD_NUMBER: _ClassVar[int]
    EXECUTE_SHARD_FIELD_NUMBER: _ClassVar[int]
    EXECUTE_PIPELINE_STAGE_FIELD_NUMBER: _ClassVar[int]
    registration_accepted: RegistrationAccepted
    registration_rejected: RegistrationRejected
    execute_task: ExecuteTask
    cancel_task: CancelTask
    heartbeat_ack: HeartbeatAck
    execute_shard: ExecuteShard
    execute_pipeline_stage: ExecutePipelineStage
    def __init__(self, registration_accepted: _Optional[_Union[RegistrationAccepted, _Mapping]] = ..., registration_rejected: _Optional[_Union[RegistrationRejected, _Mapping]] = ..., execute_task: _Optional[_Union[ExecuteTask, _Mapping]] = ..., cancel_task: _Optional[_Union[CancelTask, _Mapping]] = ..., heartbeat_ack: _Optional[_Union[HeartbeatAck, _Mapping]] = ..., execute_shard: _Optional[_Union[ExecuteShard, _Mapping]] = ..., execute_pipeline_stage: _Optional[_Union[ExecutePipelineStage, _Mapping]] = ...) -> None: ...

class RegisterDevice(_message.Message):
    __slots__ = ("device_id", "display_name", "device_type", "platform", "agent_version", "enrollment_token", "total_memory_mb", "models", "certificate_fingerprint", "hardware", "personal_profile")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    AGENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENROLLMENT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    CERTIFICATE_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    HARDWARE_FIELD_NUMBER: _ClassVar[int]
    PERSONAL_PROFILE_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    display_name: str
    device_type: str
    platform: str
    agent_version: str
    enrollment_token: str
    total_memory_mb: int
    models: _containers.RepeatedCompositeFieldContainer[ModelCapability]
    certificate_fingerprint: str
    hardware: HardwareInventory
    personal_profile: PersonalProfileRegistration
    def __init__(self, device_id: _Optional[str] = ..., display_name: _Optional[str] = ..., device_type: _Optional[str] = ..., platform: _Optional[str] = ..., agent_version: _Optional[str] = ..., enrollment_token: _Optional[str] = ..., total_memory_mb: _Optional[int] = ..., models: _Optional[_Iterable[_Union[ModelCapability, _Mapping]]] = ..., certificate_fingerprint: _Optional[str] = ..., hardware: _Optional[_Union[HardwareInventory, _Mapping]] = ..., personal_profile: _Optional[_Union[PersonalProfileRegistration, _Mapping]] = ...) -> None: ...

class PersonalProfileRegistration(_message.Message):
    __slots__ = ("person_name", "preferred_mode", "steering_vector_id", "steering_alpha", "steering_positions", "allow_remote_vector", "notes")
    PERSON_NAME_FIELD_NUMBER: _ClassVar[int]
    PREFERRED_MODE_FIELD_NUMBER: _ClassVar[int]
    STEERING_VECTOR_ID_FIELD_NUMBER: _ClassVar[int]
    STEERING_ALPHA_FIELD_NUMBER: _ClassVar[int]
    STEERING_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_REMOTE_VECTOR_FIELD_NUMBER: _ClassVar[int]
    NOTES_FIELD_NUMBER: _ClassVar[int]
    person_name: str
    preferred_mode: str
    steering_vector_id: str
    steering_alpha: float
    steering_positions: str
    allow_remote_vector: bool
    notes: str
    def __init__(self, person_name: _Optional[str] = ..., preferred_mode: _Optional[str] = ..., steering_vector_id: _Optional[str] = ..., steering_alpha: _Optional[float] = ..., steering_positions: _Optional[str] = ..., allow_remote_vector: _Optional[bool] = ..., notes: _Optional[str] = ...) -> None: ...

class HardwareInventory(_message.Message):
    __slots__ = ("manufacturer", "model", "device", "os_version", "api_level", "soc_manufacturer", "soc_model", "cpu_abis", "cpu_core_count", "total_storage_mb", "available_storage_mb", "npu_status", "npu_name", "qnn_runtime_version")
    MANUFACTURER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    OS_VERSION_FIELD_NUMBER: _ClassVar[int]
    API_LEVEL_FIELD_NUMBER: _ClassVar[int]
    SOC_MANUFACTURER_FIELD_NUMBER: _ClassVar[int]
    SOC_MODEL_FIELD_NUMBER: _ClassVar[int]
    CPU_ABIS_FIELD_NUMBER: _ClassVar[int]
    CPU_CORE_COUNT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STORAGE_MB_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_STORAGE_MB_FIELD_NUMBER: _ClassVar[int]
    NPU_STATUS_FIELD_NUMBER: _ClassVar[int]
    NPU_NAME_FIELD_NUMBER: _ClassVar[int]
    QNN_RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    manufacturer: str
    model: str
    device: str
    os_version: str
    api_level: int
    soc_manufacturer: str
    soc_model: str
    cpu_abis: _containers.RepeatedScalarFieldContainer[str]
    cpu_core_count: int
    total_storage_mb: int
    available_storage_mb: int
    npu_status: str
    npu_name: str
    qnn_runtime_version: str
    def __init__(self, manufacturer: _Optional[str] = ..., model: _Optional[str] = ..., device: _Optional[str] = ..., os_version: _Optional[str] = ..., api_level: _Optional[int] = ..., soc_manufacturer: _Optional[str] = ..., soc_model: _Optional[str] = ..., cpu_abis: _Optional[_Iterable[str]] = ..., cpu_core_count: _Optional[int] = ..., total_storage_mb: _Optional[int] = ..., available_storage_mb: _Optional[int] = ..., npu_status: _Optional[str] = ..., npu_name: _Optional[str] = ..., qnn_runtime_version: _Optional[str] = ...) -> None: ...

class ModelCapability(_message.Message):
    __slots__ = ("model_id", "model_family", "role", "task_classes", "max_context_tokens", "warm", "quality_score", "steering_vector_ids", "supported_steering_layers", "segment", "model_version", "tokenizer_id", "precision", "boundary_format", "runtime_name", "runtime_version", "supported_accelerators", "min_memory_mb", "supports_steering", "supports_data_parallel", "supports_layer_pipeline")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FAMILY_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TASK_CLASSES_FIELD_NUMBER: _ClassVar[int]
    MAX_CONTEXT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    WARM_FIELD_NUMBER: _ClassVar[int]
    QUALITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    STEERING_VECTOR_IDS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_STEERING_LAYERS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_ID_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_FORMAT_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_NAME_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ACCELERATORS_FIELD_NUMBER: _ClassVar[int]
    MIN_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_STEERING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_DATA_PARALLEL_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_LAYER_PIPELINE_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    model_family: str
    role: str
    task_classes: _containers.RepeatedScalarFieldContainer[str]
    max_context_tokens: int
    warm: bool
    quality_score: float
    steering_vector_ids: _containers.RepeatedScalarFieldContainer[str]
    supported_steering_layers: _containers.RepeatedScalarFieldContainer[int]
    segment: ModelSegment
    model_version: str
    tokenizer_id: str
    precision: str
    boundary_format: str
    runtime_name: str
    runtime_version: str
    supported_accelerators: _containers.RepeatedScalarFieldContainer[str]
    min_memory_mb: int
    supports_steering: bool
    supports_data_parallel: bool
    supports_layer_pipeline: bool
    def __init__(self, model_id: _Optional[str] = ..., model_family: _Optional[str] = ..., role: _Optional[str] = ..., task_classes: _Optional[_Iterable[str]] = ..., max_context_tokens: _Optional[int] = ..., warm: _Optional[bool] = ..., quality_score: _Optional[float] = ..., steering_vector_ids: _Optional[_Iterable[str]] = ..., supported_steering_layers: _Optional[_Iterable[int]] = ..., segment: _Optional[_Union[ModelSegment, _Mapping]] = ..., model_version: _Optional[str] = ..., tokenizer_id: _Optional[str] = ..., precision: _Optional[str] = ..., boundary_format: _Optional[str] = ..., runtime_name: _Optional[str] = ..., runtime_version: _Optional[str] = ..., supported_accelerators: _Optional[_Iterable[str]] = ..., min_memory_mb: _Optional[int] = ..., supports_steering: _Optional[bool] = ..., supports_data_parallel: _Optional[bool] = ..., supports_layer_pipeline: _Optional[bool] = ...) -> None: ...

class ModelSegment(_message.Message):
    __slots__ = ("pipeline_id", "start_layer", "end_layer", "total_layers", "includes_embedding", "includes_lm_head")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LAYERS_FIELD_NUMBER: _ClassVar[int]
    INCLUDES_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    INCLUDES_LM_HEAD_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    start_layer: int
    end_layer: int
    total_layers: int
    includes_embedding: bool
    includes_lm_head: bool
    def __init__(self, pipeline_id: _Optional[str] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., total_layers: _Optional[int] = ..., includes_embedding: _Optional[bool] = ..., includes_lm_head: _Optional[bool] = ...) -> None: ...

class HealthUpdate(_message.Message):
    __slots__ = ("device_id", "timestamp_ms", "battery_pct", "charging", "thermal_level", "cpu_utilization", "accelerator_utilization", "available_memory_mb", "network_rtt_ms", "reachable", "active_task_ids", "simulated_constraint", "warm_model_ids")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    BATTERY_PCT_FIELD_NUMBER: _ClassVar[int]
    CHARGING_FIELD_NUMBER: _ClassVar[int]
    THERMAL_LEVEL_FIELD_NUMBER: _ClassVar[int]
    CPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    NETWORK_RTT_MS_FIELD_NUMBER: _ClassVar[int]
    REACHABLE_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_TASK_IDS_FIELD_NUMBER: _ClassVar[int]
    SIMULATED_CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
    WARM_MODEL_IDS_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    timestamp_ms: int
    battery_pct: float
    charging: bool
    thermal_level: float
    cpu_utilization: float
    accelerator_utilization: float
    available_memory_mb: int
    network_rtt_ms: float
    reachable: bool
    active_task_ids: _containers.RepeatedScalarFieldContainer[str]
    simulated_constraint: bool
    warm_model_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, device_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., battery_pct: _Optional[float] = ..., charging: _Optional[bool] = ..., thermal_level: _Optional[float] = ..., cpu_utilization: _Optional[float] = ..., accelerator_utilization: _Optional[float] = ..., available_memory_mb: _Optional[int] = ..., network_rtt_ms: _Optional[float] = ..., reachable: _Optional[bool] = ..., active_task_ids: _Optional[_Iterable[str]] = ..., simulated_constraint: _Optional[bool] = ..., warm_model_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ExecuteTask(_message.Message):
    __slots__ = ("task_id", "attempt_id", "request_text", "model_id", "timeout_ms", "steering")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    STEERING_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    request_text: str
    model_id: str
    timeout_ms: int
    steering: SteeringSpec
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., request_text: _Optional[str] = ..., model_id: _Optional[str] = ..., timeout_ms: _Optional[int] = ..., steering: _Optional[_Union[SteeringSpec, _Mapping]] = ...) -> None: ...

class ExecuteShard(_message.Message):
    __slots__ = ("task_id", "attempt_id", "shard_id", "request_text", "model_id", "timeout_ms", "steering")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    STEERING_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    shard_id: str
    request_text: str
    model_id: str
    timeout_ms: int
    steering: SteeringSpec
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., shard_id: _Optional[str] = ..., request_text: _Optional[str] = ..., model_id: _Optional[str] = ..., timeout_ms: _Optional[int] = ..., steering: _Optional[_Union[SteeringSpec, _Mapping]] = ...) -> None: ...

class TaskResult(_message.Message):
    __slots__ = ("task_id", "attempt_id", "device_id", "success", "output_text", "error_code", "error_message", "metrics")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    device_id: str
    success: bool
    output_text: str
    error_code: str
    error_message: str
    metrics: ExecutionMetrics
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., device_id: _Optional[str] = ..., success: _Optional[bool] = ..., output_text: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., metrics: _Optional[_Union[ExecutionMetrics, _Mapping]] = ...) -> None: ...

class PartialTaskResult(_message.Message):
    __slots__ = ("task_id", "attempt_id", "shard_id", "device_id", "success", "output_text", "error_code", "error_message", "metrics")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    shard_id: str
    device_id: str
    success: bool
    output_text: str
    error_code: str
    error_message: str
    metrics: ExecutionMetrics
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., shard_id: _Optional[str] = ..., device_id: _Optional[str] = ..., success: _Optional[bool] = ..., output_text: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., metrics: _Optional[_Union[ExecutionMetrics, _Mapping]] = ...) -> None: ...

class BoundaryTensor(_message.Message):
    __slots__ = ("tensor_name", "dtype", "shape", "data", "checksum")
    TENSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    tensor_name: str
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    checksum: str
    def __init__(self, tensor_name: _Optional[str] = ..., dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., checksum: _Optional[str] = ...) -> None: ...

class ExecutePipelineStage(_message.Message):
    __slots__ = ("task_id", "attempt_id", "stage_id", "stage_index", "request_text", "model_id", "input_boundary", "final_stage", "timeout_ms", "steering")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    FINAL_STAGE_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    STEERING_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    stage_id: str
    stage_index: int
    request_text: str
    model_id: str
    input_boundary: BoundaryTensor
    final_stage: bool
    timeout_ms: int
    steering: SteeringSpec
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., stage_id: _Optional[str] = ..., stage_index: _Optional[int] = ..., request_text: _Optional[str] = ..., model_id: _Optional[str] = ..., input_boundary: _Optional[_Union[BoundaryTensor, _Mapping]] = ..., final_stage: _Optional[bool] = ..., timeout_ms: _Optional[int] = ..., steering: _Optional[_Union[SteeringSpec, _Mapping]] = ...) -> None: ...

class PipelineStageResult(_message.Message):
    __slots__ = ("task_id", "attempt_id", "stage_id", "device_id", "success", "output_boundary", "output_text", "error_code", "error_message", "metrics")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    stage_id: str
    device_id: str
    success: bool
    output_boundary: BoundaryTensor
    output_text: str
    error_code: str
    error_message: str
    metrics: ExecutionMetrics
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., stage_id: _Optional[str] = ..., device_id: _Optional[str] = ..., success: _Optional[bool] = ..., output_boundary: _Optional[_Union[BoundaryTensor, _Mapping]] = ..., output_text: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., metrics: _Optional[_Union[ExecutionMetrics, _Mapping]] = ...) -> None: ...

class ExecutionMetrics(_message.Message):
    __slots__ = ("model_id", "model_version", "runtime_name", "runtime_version", "accelerator", "execution_latency_ms", "error_code", "error_message", "observed_memory_delta_mb", "observed_thermal_delta")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_NAME_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_MEMORY_DELTA_MB_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_THERMAL_DELTA_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    model_version: str
    runtime_name: str
    runtime_version: str
    accelerator: str
    execution_latency_ms: int
    error_code: str
    error_message: str
    observed_memory_delta_mb: int
    observed_thermal_delta: float
    def __init__(self, model_id: _Optional[str] = ..., model_version: _Optional[str] = ..., runtime_name: _Optional[str] = ..., runtime_version: _Optional[str] = ..., accelerator: _Optional[str] = ..., execution_latency_ms: _Optional[int] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., observed_memory_delta_mb: _Optional[int] = ..., observed_thermal_delta: _Optional[float] = ...) -> None: ...

class SteeringSpec(_message.Message):
    __slots__ = ("enabled", "vector_id", "model_family", "target_layer", "alpha", "positions", "allow_remote_vector")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    VECTOR_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FAMILY_FIELD_NUMBER: _ClassVar[int]
    TARGET_LAYER_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_REMOTE_VECTOR_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    vector_id: str
    model_family: str
    target_layer: int
    alpha: float
    positions: str
    allow_remote_vector: bool
    def __init__(self, enabled: _Optional[bool] = ..., vector_id: _Optional[str] = ..., model_family: _Optional[str] = ..., target_layer: _Optional[int] = ..., alpha: _Optional[float] = ..., positions: _Optional[str] = ..., allow_remote_vector: _Optional[bool] = ...) -> None: ...

class ShutdownEvent(_message.Message):
    __slots__ = ("device_id", "reason", "simulated")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SIMULATED_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    reason: str
    simulated: bool
    def __init__(self, device_id: _Optional[str] = ..., reason: _Optional[str] = ..., simulated: _Optional[bool] = ...) -> None: ...

class RegistrationAccepted(_message.Message):
    __slots__ = ("brain_id", "heartbeat_interval_ms", "device_credential")
    BRAIN_ID_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    brain_id: str
    heartbeat_interval_ms: int
    device_credential: str
    def __init__(self, brain_id: _Optional[str] = ..., heartbeat_interval_ms: _Optional[int] = ..., device_credential: _Optional[str] = ...) -> None: ...

class RegistrationRejected(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class CancelTask(_message.Message):
    __slots__ = ("task_id", "attempt_id", "reason")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    attempt_id: str
    reason: str
    def __init__(self, task_id: _Optional[str] = ..., attempt_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class HeartbeatAck(_message.Message):
    __slots__ = ("brain_timestamp_ms",)
    BRAIN_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    brain_timestamp_ms: int
    def __init__(self, brain_timestamp_ms: _Optional[int] = ...) -> None: ...

class SubmitTaskRequest(_message.Message):
    __slots__ = ("request_text", "preferred_mode", "execution_mode", "timeout_ms", "steering", "origin_device_id", "reducer", "use_profile_steering")
    REQUEST_TEXT_FIELD_NUMBER: _ClassVar[int]
    PREFERRED_MODE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_MODE_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    STEERING_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    REDUCER_FIELD_NUMBER: _ClassVar[int]
    USE_PROFILE_STEERING_FIELD_NUMBER: _ClassVar[int]
    request_text: str
    preferred_mode: str
    execution_mode: str
    timeout_ms: int
    steering: SteeringSpec
    origin_device_id: str
    reducer: str
    use_profile_steering: bool
    def __init__(self, request_text: _Optional[str] = ..., preferred_mode: _Optional[str] = ..., execution_mode: _Optional[str] = ..., timeout_ms: _Optional[int] = ..., steering: _Optional[_Union[SteeringSpec, _Mapping]] = ..., origin_device_id: _Optional[str] = ..., reducer: _Optional[str] = ..., use_profile_steering: _Optional[bool] = ...) -> None: ...

class SubmitTaskResponse(_message.Message):
    __slots__ = ("task_id", "state", "success", "output_text", "error_code", "error_message", "accepted_attempt_id", "device_id", "model_id", "route_reasons", "steering", "origin_device_id", "reducer")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TEXT_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_ATTEMPT_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ROUTE_REASONS_FIELD_NUMBER: _ClassVar[int]
    STEERING_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    REDUCER_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    state: str
    success: bool
    output_text: str
    error_code: str
    error_message: str
    accepted_attempt_id: str
    device_id: str
    model_id: str
    route_reasons: _containers.RepeatedScalarFieldContainer[str]
    steering: SteeringSpec
    origin_device_id: str
    reducer: str
    def __init__(self, task_id: _Optional[str] = ..., state: _Optional[str] = ..., success: _Optional[bool] = ..., output_text: _Optional[str] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., accepted_attempt_id: _Optional[str] = ..., device_id: _Optional[str] = ..., model_id: _Optional[str] = ..., route_reasons: _Optional[_Iterable[str]] = ..., steering: _Optional[_Union[SteeringSpec, _Mapping]] = ..., origin_device_id: _Optional[str] = ..., reducer: _Optional[str] = ...) -> None: ...

class GetTaskRequest(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    def __init__(self, task_id: _Optional[str] = ...) -> None: ...
