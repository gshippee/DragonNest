"""Physical Qwen3-1.7B stage-local sliding-window KV state."""

from __future__ import annotations

import numpy as np

CONTEXT_LENGTH = 512


class StageKVBuffer:
    """Own quantized KV for absolute transformer-layer indices.

    Physical X Elite execution proved ``past_*_out`` is a newly-computed
    delta. Decode therefore appends it to the existing window and retains the
    newest 512 positions; it never replaces the 511-token history.
    """

    def __init__(self, layer_indices: list[int], zero_point: int = 128):
        self.layer_indices = list(layer_indices)
        self.key = {
            layer: np.full(
                (8, 1, 128, CONTEXT_LENGTH), zero_point, dtype=np.uint8
            )
            for layer in layer_indices
        }
        self.value = {
            layer: np.full(
                (8, 1, CONTEXT_LENGTH, 128), zero_point, dtype=np.uint8
            )
            for layer in layer_indices
        }

    def get_past_in(
        self, layer: int, past_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.key[layer][:, :, :, -past_len:],
            self.value[layer][:, :, -past_len:, :],
        )

    def update(
        self,
        layer: int,
        new_key_delta: np.ndarray,
        new_value_delta: np.ndarray,
    ) -> None:
        self.key[layer] = np.concatenate(
            [self.key[layer], new_key_delta], axis=3
        )[:, :, :, -CONTEXT_LENGTH:]
        self.value[layer] = np.concatenate(
            [self.value[layer], new_value_delta], axis=2
        )[:, :, -CONTEXT_LENGTH:, :]

    def seed_from_prompt(
        self,
        layer: int,
        prompt_past_in: np.ndarray,
        prompt_delta: np.ndarray,
        axis_is_key: bool,
    ) -> None:
        axis = 3 if axis_is_key else 2
        full = np.concatenate([prompt_past_in, prompt_delta], axis=axis)
        if full.shape[axis] != CONTEXT_LENGTH:
            raise ValueError(
                "prompt KV seed must contain exactly "
                f"{CONTEXT_LENGTH} positions, got {full.shape[axis]}"
            )
        if axis_is_key:
            self.key[layer] = full
        else:
            self.value[layer] = full

    def release(self) -> None:
        self.key.clear()
        self.value.clear()
