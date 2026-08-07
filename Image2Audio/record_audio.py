"""Push-to-talk microphone recorder: press Enter to start, press Enter again
to stop, write the captured audio to a WAV file via soundfile -- the same
library every other pipeline in this repo already uses for audio I/O, so the
output is a drop-in input to whisper_pipeline.run()/whisper_onnx_pipeline.run().

No mic-capture code existed anywhere in this repo before this file; sounddevice
is a new dependency (not previously installed in qai_env)."""

from __future__ import annotations

import platform
import queue

import numpy as np
import soundfile as sf

# qai_env's python.exe is actually x64 (confirmed via its PE header), but
# platform.machine() reports this machine's native "ARM64" regardless of the
# process's own emulated architecture. sounddevice picks its bundled
# PortAudio DLL from platform.machine() and, for "arm64", looks for
# libportaudioarm64.dll -- which this sounddevice version's bundled binaries
# don't include (only libportaudio64bit.dll/-asio.dll, i.e. x64). Spoofing
# platform.machine() during the import only steers sounddevice to the x64
# DLL that actually matches this process and that does ship in the package.
_real_machine = platform.machine
platform.machine = lambda: "AMD64"
try:
    import sounddevice as sd
finally:
    platform.machine = _real_machine


def record_to_wav(output_path: str, sample_rate: int = 16000, channels: int = 1) -> str:
    input("Press Enter to start recording...")

    frames: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, frame_count, time_info, status) -> None:
        frames.put(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", callback=_callback):
        print("Recording... press Enter to stop.")
        input()

    chunks = []
    while not frames.empty():
        chunks.append(frames.get())
    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, channels), dtype=np.float32)

    sf.write(output_path, audio, samplerate=sample_rate)
    print(f"Saved {audio.shape[0] / sample_rate:.1f}s of audio to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    record_to_wav(args.output_wav, sample_rate=args.sample_rate)
