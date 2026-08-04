from __future__ import annotations

from ..models import (
    Device,
    ExecutionMetrics,
    HardwareInventory,
    HealthState,
    ModelCapability,
    ModelSegment,
    SteeringSpec,
    TaskResult,
)
from ..proto import dragonnest_pb2 as pb


def device_from_registration(message: pb.RegisterDevice) -> Device:
    models = []
    for model in message.models:
        segment = None
        if model.HasField("segment") and model.segment.pipeline_id:
            segment = ModelSegment(
                pipeline_id=model.segment.pipeline_id,
                start_layer=model.segment.start_layer,
                end_layer=model.segment.end_layer,
                total_layers=model.segment.total_layers,
                includes_embedding=model.segment.includes_embedding,
                includes_lm_head=model.segment.includes_lm_head,
            )
        models.append(
            ModelCapability(
                model_id=model.model_id,
                model_family=model.model_family,
                role=model.role,
                task_classes=tuple(model.task_classes),
                max_context_tokens=model.max_context_tokens,
                warm=model.warm,
                quality_score=model.quality_score,
                model_version=model.model_version,
                tokenizer_id=model.tokenizer_id,
                precision=model.precision,
                boundary_format=model.boundary_format,
                steering_vector_ids=tuple(model.steering_vector_ids),
                supported_steering_layers=tuple(model.supported_steering_layers),
                segment=segment,
                runtime_name=model.runtime_name,
                runtime_version=model.runtime_version,
                supported_accelerators=tuple(model.supported_accelerators),
                min_memory_mb=model.min_memory_mb,
                supports_steering=model.supports_steering,
                supports_data_parallel=model.supports_data_parallel,
                supports_layer_pipeline=model.supports_layer_pipeline,
            )
        )
    return Device(
        device_id=message.device_id,
        display_name=message.display_name,
        device_type=message.device_type,
        platform=message.platform,
        total_memory_mb=message.total_memory_mb,
        health=HealthState(
            battery_pct=-1,
            available_memory_mb=message.total_memory_mb,
            reachable=True,
        ),
        models=tuple(models),
        hardware=HardwareInventory(
            manufacturer=message.hardware.manufacturer,
            model=message.hardware.model,
            device=message.hardware.device,
            os_version=message.hardware.os_version,
            api_level=message.hardware.api_level,
            soc_manufacturer=message.hardware.soc_manufacturer,
            soc_model=message.hardware.soc_model,
            cpu_abis=tuple(message.hardware.cpu_abis),
            cpu_core_count=message.hardware.cpu_core_count,
            total_storage_mb=message.hardware.total_storage_mb,
            available_storage_mb=message.hardware.available_storage_mb,
            npu_status=message.hardware.npu_status or "not_probed",
            npu_name=message.hardware.npu_name,
            qnn_runtime_version=message.hardware.qnn_runtime_version,
        ),
    )


def registration_from_device(
    device: Device,
    enrollment_token: str,
    agent_version: str,
    certificate_fingerprint: str = "",
) -> pb.RegisterDevice:
    models = []
    for model in device.models:
        segment = None
        if model.segment:
            segment = pb.ModelSegment(
                pipeline_id=model.segment.pipeline_id,
                start_layer=model.segment.start_layer,
                end_layer=model.segment.end_layer,
                total_layers=model.segment.total_layers,
                includes_embedding=model.segment.includes_embedding,
                includes_lm_head=model.segment.includes_lm_head,
            )
        models.append(
            pb.ModelCapability(
                model_id=model.model_id,
                model_family=model.model_family,
                role=model.role,
                task_classes=model.task_classes,
                max_context_tokens=model.max_context_tokens,
                warm=model.warm,
                quality_score=model.quality_score,
                model_version=model.model_version,
                tokenizer_id=model.tokenizer_id,
                precision=model.precision,
                boundary_format=model.boundary_format,
                steering_vector_ids=model.steering_vector_ids,
                supported_steering_layers=model.supported_steering_layers,
                segment=segment,
                runtime_name=model.runtime_name,
                runtime_version=model.runtime_version,
                supported_accelerators=model.supported_accelerators,
                min_memory_mb=model.min_memory_mb,
                supports_steering=model.supports_steering,
                supports_data_parallel=model.supports_data_parallel,
                supports_layer_pipeline=model.supports_layer_pipeline,
            )
        )
    return pb.RegisterDevice(
        device_id=device.device_id,
        display_name=device.display_name,
        device_type=device.device_type,
        platform=device.platform,
        agent_version=agent_version,
        enrollment_token=enrollment_token,
        total_memory_mb=device.total_memory_mb,
        models=models,
        certificate_fingerprint=certificate_fingerprint,
        hardware=pb.HardwareInventory(
            manufacturer=device.hardware.manufacturer,
            model=device.hardware.model,
            device=device.hardware.device,
            os_version=device.hardware.os_version,
            api_level=device.hardware.api_level,
            soc_manufacturer=device.hardware.soc_manufacturer,
            soc_model=device.hardware.soc_model,
            cpu_abis=device.hardware.cpu_abis,
            cpu_core_count=device.hardware.cpu_core_count,
            total_storage_mb=device.hardware.total_storage_mb,
            available_storage_mb=device.hardware.available_storage_mb,
            npu_status=device.hardware.npu_status,
            npu_name=device.hardware.npu_name,
            qnn_runtime_version=device.hardware.qnn_runtime_version,
        ),
    )


def health_from_proto(message: pb.HealthUpdate) -> HealthState:
    return HealthState(
        battery_pct=message.battery_pct,
        charging=message.charging,
        thermal_level=message.thermal_level,
        cpu_utilization=message.cpu_utilization,
        accelerator_utilization=message.accelerator_utilization,
        available_memory_mb=message.available_memory_mb,
        network_rtt_ms=message.network_rtt_ms,
        reachable=message.reachable,
    )


def health_to_proto(device: Device, timestamp_ms: int) -> pb.HealthUpdate:
    health = device.health
    return pb.HealthUpdate(
        device_id=device.device_id,
        timestamp_ms=timestamp_ms,
        battery_pct=health.battery_pct,
        charging=health.charging,
        thermal_level=health.thermal_level,
        cpu_utilization=health.cpu_utilization,
        accelerator_utilization=health.accelerator_utilization,
        available_memory_mb=health.available_memory_mb,
        network_rtt_ms=health.network_rtt_ms,
        reachable=health.reachable,
    )


def task_result_from_proto(message: pb.TaskResult) -> TaskResult:
    metrics = None
    if message.HasField("metrics") and message.metrics.model_id:
        metrics = ExecutionMetrics(
            model_id=message.metrics.model_id,
            model_version=message.metrics.model_version,
            runtime_name=message.metrics.runtime_name,
            runtime_version=message.metrics.runtime_version,
            accelerator=message.metrics.accelerator,
            execution_latency_ms=message.metrics.execution_latency_ms,
            error_code=message.metrics.error_code,
            error_message=message.metrics.error_message,
            observed_memory_delta_mb=message.metrics.observed_memory_delta_mb,
            observed_thermal_delta=message.metrics.observed_thermal_delta,
        )
    return TaskResult(
        task_id=message.task_id,
        attempt_id=message.attempt_id,
        success=message.success,
        output_text=message.output_text,
        device_id=message.device_id,
        error_code=message.error_code,
        error_message=message.error_message,
        latency_ms=metrics.execution_latency_ms if metrics else 0,
        metrics=metrics,
    )


def task_result_to_proto(result: TaskResult) -> pb.TaskResult:
    metrics = None
    if result.metrics:
        metrics = pb.ExecutionMetrics(
            model_id=result.metrics.model_id,
            model_version=result.metrics.model_version,
            runtime_name=result.metrics.runtime_name,
            runtime_version=result.metrics.runtime_version,
            accelerator=result.metrics.accelerator,
            execution_latency_ms=result.metrics.execution_latency_ms,
            error_code=result.metrics.error_code,
            error_message=result.metrics.error_message,
            observed_memory_delta_mb=result.metrics.observed_memory_delta_mb or 0,
            observed_thermal_delta=result.metrics.observed_thermal_delta or 0,
        )
    return pb.TaskResult(
        task_id=result.task_id,
        attempt_id=result.attempt_id,
        device_id=result.device_id,
        success=result.success,
        output_text=result.output_text,
        error_code=result.error_code,
        error_message=result.error_message,
        metrics=metrics,
    )


def partial_result_from_proto(message: pb.PartialTaskResult) -> TaskResult:
    return task_result_from_proto(
        pb.TaskResult(
            task_id=message.task_id,
            attempt_id=message.attempt_id,
            device_id=message.device_id,
            success=message.success,
            output_text=message.output_text,
            error_code=message.error_code,
            error_message=message.error_message,
            metrics=message.metrics,
        )
    )


def partial_result_to_proto(result: TaskResult, shard_id: str) -> pb.PartialTaskResult:
    task_message = task_result_to_proto(result)
    return pb.PartialTaskResult(
        task_id=task_message.task_id,
        attempt_id=task_message.attempt_id,
        shard_id=shard_id,
        device_id=task_message.device_id,
        success=task_message.success,
        output_text=task_message.output_text,
        error_code=task_message.error_code,
        error_message=task_message.error_message,
        metrics=task_message.metrics,
    )


def pipeline_result_from_proto(message: pb.PipelineStageResult) -> TaskResult:
    return task_result_from_proto(
        pb.TaskResult(
            task_id=message.task_id,
            attempt_id=message.attempt_id,
            device_id=message.device_id,
            success=message.success,
            output_text=message.output_text,
            error_code=message.error_code,
            error_message=message.error_message,
            metrics=message.metrics,
        )
    )


def steering_from_proto(message: pb.SteeringSpec) -> SteeringSpec:
    return SteeringSpec(
        enabled=message.enabled,
        vector_id=message.vector_id,
        model_family=message.model_family,
        target_layer=message.target_layer,
        alpha=message.alpha,
        positions=message.positions or "last",
        allow_remote_vector=message.allow_remote_vector,
    )


def steering_to_proto(spec: SteeringSpec) -> pb.SteeringSpec:
    return pb.SteeringSpec(
        enabled=spec.enabled,
        vector_id=spec.vector_id,
        model_family=spec.model_family,
        target_layer=spec.target_layer,
        alpha=spec.alpha,
        positions=spec.positions,
        allow_remote_vector=spec.allow_remote_vector,
    )
