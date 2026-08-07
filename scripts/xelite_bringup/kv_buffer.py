"""Sliding-window per-stage KV buffer for the Qwen3-1.7B X Elite split
pipeline.

Physical execution on a Snapdragon X Elite (see
docs/results/qwen3_1_7b_xelite_physical_bringup.md) confirmed the compiled
QNN graphs' `past_key_*_out` / `past_value_*_out` tensors are a DELTA --
only the newly-computed positions -- never a full replacement of the KV
cache. This models the buffer a real inference engine maintains: a fixed
`CONTEXT_LENGTH`-wide sliding window per transformer layer, updated by
concatenating each call's real input past with its real output delta, then
keeping only the most recent `CONTEXT_LENGTH` entries. A prompt call
(`past_len=384`, delta 128) and a decode call (`past_len=511`, delta 1) both
satisfy the same update-then-slide rule -- no special-casing needed.
"""
import numpy as np

CONTEXT_LENGTH = 512


class StageKVBuffer:
    """Owns the sliding-window KV state for one pipeline stage's transformer
    layers. `layer_indices` must be the ABSOLUTE transformer-layer indices
    used in that stage's compiled graph's tensor names (e.g. stage 2 uses
    `past_key_10_in`..`past_key_19_in`, not a stage-relative 0-9) -- confirmed
    per stage via `qnn-context-binary-utility.exe`'s JSON dump, never
    assumed from another stage's naming."""

    def __init__(self, layer_indices: list[int], zero_point: int = 128):
        self.layer_indices = layer_indices
        # [8,1,128,512] key / [8,1,512,128] value per layer, filled at the
        # tensor's own quantized zero point (i.e. real value 0.0).
        self.key = {n: np.full((8, 1, 128, CONTEXT_LENGTH), zero_point, dtype=np.uint8) for n in layer_indices}
        self.value = {n: np.full((8, 1, CONTEXT_LENGTH, 128), zero_point, dtype=np.uint8) for n in layer_indices}

    def get_past_in(self, layer: int, past_len: int) -> tuple[np.ndarray, np.ndarray]:
        return self.key[layer][:, :, :, -past_len:], self.value[layer][:, :, -past_len:, :]

    def update(self, layer: int, new_key_delta: np.ndarray, new_value_delta: np.ndarray) -> None:
        self.key[layer] = np.concatenate([self.key[layer], new_key_delta], axis=3)[:, :, :, -CONTEXT_LENGTH:]
        self.value[layer] = np.concatenate([self.value[layer], new_value_delta], axis=2)[:, :, -CONTEXT_LENGTH:, :]

    def seed_from_prompt(
        self, layer: int, prompt_past_in: np.ndarray, prompt_delta: np.ndarray, axis_is_key: bool
    ) -> None:
        """Seed the buffer directly from a completed prompt call's own
        past_*_in (zero) + past_*_out (real delta) -- equivalent to update()
        starting from an all-zero buffer, but avoids re-materializing the
        zero prefix. `prompt_past_in.shape[axis] + prompt_delta.shape[axis]`
        must equal CONTEXT_LENGTH (384 + 128 = 512 for this pipeline)."""
        axis = 3 if axis_is_key else 2
        full = np.concatenate([prompt_past_in, prompt_delta], axis=axis)
        assert full.shape[axis] == CONTEXT_LENGTH, (
            f"seed_from_prompt: expected concatenated length {CONTEXT_LENGTH}, got {full.shape[axis]}"
        )
        if axis_is_key:
            self.key[layer] = full
        else:
            self.value[layer] = full

    def release(self) -> None:
        self.key.clear()
        self.value.clear()
