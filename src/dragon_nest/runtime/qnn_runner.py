"""Run DLC and context-binary models with Qualcomm ``qnn-net-run``.

Ported from ``PersonaCare/qnn_runner.py``, where the implementation was
validated on a Snapdragon X Elite. DragonNest keeps the tensor I/O, retry, and
profiling behavior while making SDK and scratch locations configurable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

QAIRT_ROOT = Path(
    os.environ.get(
        "QAIRT_ROOT",
        str(Path.home() / "Downloads" / "v2.48.40.260702" / "qairt" / "2.48.40.260702"),
    )
)
ARCH = os.environ.get("QAIRT_ARCH", "aarch64-windows-msvc")
BIN_DIR = QAIRT_ROOT / "bin" / ARCH
LIB_DIR = QAIRT_ROOT / "lib" / ARCH
HEXAGON_DIR = (
    QAIRT_ROOT / "lib" / os.environ.get("QAIRT_HEXAGON_VERSION", "hexagon-v73")
)
QNN_NET_RUN = Path(os.environ.get("QNN_NET_RUN", str(BIN_DIR / "qnn-net-run.exe")))
QNN_PROFILE_VIEWER = Path(
    os.environ.get("QNN_PROFILE_VIEWER", str(BIN_DIR / "qnn-profile-viewer.exe"))
)

BACKENDS = {
    "cpu": Path(os.environ.get("QNN_CPU_BACKEND", str(LIB_DIR / "QnnCpu.dll"))),
    "htp": Path(os.environ.get("QNN_HTP_BACKEND", str(LIB_DIR / "QnnHtp.dll"))),
}

_SCRATCH_ROOT = (
    Path(
        os.environ.get(
            "DRAGONNEST_SCRATCH_DIR",
            str(Path(tempfile.gettempdir()) / "dragon_nest"),
        )
    )
    / "qnn"
)

# Real single-graph inference (detector/recognizer/encoder/flow/decoder passes
# used by this pipeline) completes in low tens of seconds even on CPU backend.
# A run past this is symptomatic of NPU/DSP session contention (e.g. a stale
# process still holding the HTP device) rather than genuine slowness -- fail
# fast with a clear error instead of hanging silently for however long the
# caller is willing to wait.
DEFAULT_TIMEOUT_SEC = 120

# qnn-net-run.exe's DSP session open (DspTransport.openSession) has been
# observed to fail intermittently -- logs "qnn_open failed, 0x80000406"
# within ~100ms and then hangs until our timeout fires -- even as the very
# first HTP call in an otherwise-idle process with nothing else contending
# for the device. It isn't contention, it's just flaky, and a fresh-process
# retry reliably gets past it. So retry the whole subprocess call a few times
# before giving up.
MAX_ATTEMPTS = 3

# When set (via enable_profiling()), every _run() call adds --profiling_level
# and copies its profiling log + CSV into this directory instead of deleting
# its scratch dir. One subfolder per call, numbered in call order.
PROFILE_DIR: Path | None = None
_profile_level = "detailed"
_profile_call_count = 0

# When set (via enable_layer_dump()), every _run() call against a --dlc_path
# graph (run_dlc() only -- context binaries can't expose intermediate tensors)
# adds --debug and copies every intermediate tensor .raw file into this
# directory, one subfolder per call.
DEBUG_DIR: Path | None = None
_debug_call_count = 0


def enable_profiling(output_dir: str | Path, level: str = "detailed") -> None:
    """Turn on per-call QNN profiling. Every subsequent run_dlc()/run_context_binary()
    call will save its profiling log (and a CSV rendering of it) under output_dir,
    one subfolder per call: 0000_<graph-file-name>/, 0001_<graph-file-name>/, ...
    Call disable_profiling() to turn this back off."""
    global PROFILE_DIR, _profile_level, _profile_call_count
    PROFILE_DIR = Path(output_dir)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    _profile_level = level
    _profile_call_count = 0


def disable_profiling() -> None:
    global PROFILE_DIR
    PROFILE_DIR = None


def enable_layer_dump(output_dir: str | Path) -> None:
    """Turn on per-call intermediate-tensor dumping (--debug). Only takes effect
    for run_dlc() calls (.dlc graphs loaded via --dlc_path) -- QNN context
    binaries (run_context_binary(), e.g. MeloTTS's .bin files) are already
    finalized and cannot expose intermediate tensors, so --debug is skipped for
    those. Every affected call saves all of its layers' raw output tensors
    under output_dir, one subfolder per call: 0000_<graph-file-name>/, ...
    Call disable_layer_dump() to turn this back off."""
    global DEBUG_DIR, _debug_call_count
    DEBUG_DIR = Path(output_dir)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    _debug_call_count = 0


def disable_layer_dump() -> None:
    global DEBUG_DIR
    DEBUG_DIR = None


def _env():
    import os

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(LIB_DIR), str(HEXAGON_DIR), env.get("PATH", "")))
    # Without this, qnn-net-run.exe can still *load* the unsigned HTP skel
    # (setSkelLogLevel succeeds) but graph preparation on the DSP silently
    # fails -- HtpTransport::graphPrepareDsp returns a null graph handle --
    # and the process then crashes (0xC0000409) trying to use it, instead of
    # returning a clean QNN error. qnn-platform-validator.exe's own DSP
    # smoke test fails the same way with an explicit "Please use testsig if
    # using unsigned images. Also make sure ADSP_LIBRARY_PATH points to
    # directory containing skels" message. Reproduced and fixed physically
    # on a Snapdragon X Elite; see
    # docs/results/qwen3_1_7b_xelite_physical_bringup.md.
    env["ADSP_LIBRARY_PATH"] = os.pathsep.join(
        (str(HEXAGON_DIR / "unsigned"), env.get("ADSP_LIBRARY_PATH", ""))
    )
    return env


def _write_raw(path: Path, array: np.ndarray) -> None:
    np.ascontiguousarray(array).tofile(path)


def _save_profile(model_flag: list[str], output_dir: Path) -> None:
    """Copy this call's profiling log out of the (about-to-be-deleted) scratch
    dir into PROFILE_DIR, and render it to CSV via qnn-profile-viewer.exe."""
    global _profile_call_count

    log_path = output_dir / "qnn-profiling-data_0.log"
    if not log_path.exists():
        return

    graph_name = Path(model_flag[-1]).stem
    dest = PROFILE_DIR / f"{_profile_call_count:04d}_{graph_name}"
    dest.mkdir(parents=True, exist_ok=True)
    _profile_call_count += 1

    shutil.copy2(log_path, dest / "qnn-profiling-data_0.log")

    csv_path = dest / "profile.csv"
    subprocess.run(
        [
            str(QNN_PROFILE_VIEWER),
            "--input_log",
            str(log_path),
            "--output",
            str(csv_path),
        ],
        cwd=output_dir,
        env=_env(),
        capture_output=True,
        text=True,
    )
    text_path = dest / "profile.txt"
    text_result = subprocess.run(
        [str(QNN_PROFILE_VIEWER), "--input_log", str(log_path)],
        cwd=output_dir,
        env=_env(),
        capture_output=True,
        text=True,
    )
    text_path.write_text(text_result.stdout)


def _save_layer_dump(
    model_flag: list[str], result_dir: Path, item_index: int = 0
) -> None:
    """Copy every intermediate-tensor .raw file this --debug call produced out
    of the (about-to-be-deleted) scratch dir into DEBUG_DIR."""
    global _debug_call_count

    raw_files = sorted(result_dir.glob("*.raw"))
    if not raw_files:
        return

    graph_name = Path(model_flag[-1]).stem
    dest = DEBUG_DIR / f"{_debug_call_count:04d}_{graph_name}_item{item_index}"
    dest.mkdir(parents=True, exist_ok=True)
    _debug_call_count += 1

    for raw_file in raw_files:
        shutil.copy2(raw_file, dest / raw_file.name)


def run_dlc(
    dlc_path: str | Path,
    inputs: dict[str, np.ndarray],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, np.ndarray]:
    """Run a .dlc graph (loaded via --dlc_path) on one input and return its outputs."""
    return _run(
        model_flag=["--dlc_path", str(dlc_path)],
        inputs=inputs,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend=backend,
        timeout_sec=timeout_sec,
    )


def run_dlc_batch(
    dlc_path: str | Path,
    inputs_list: list[dict[str, np.ndarray]],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> list[dict[str, np.ndarray]]:
    """Run a .dlc graph on many inputs within a single qnn-net-run.exe process
    (one line per input in input_list.txt) instead of one process per input.
    Cuts N per-call process-spawn + graph-load + DSP-session-open overheads
    down to one -- the dominant cost when a caller (e.g. EasyOCR's recognizer,
    called once per detected text box) would otherwise invoke run_dlc() in a
    loop."""
    return _run_batch(
        model_flag=["--dlc_path", str(dlc_path)],
        inputs_list=inputs_list,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend=backend,
        timeout_sec=timeout_sec,
    )


def run_context_binary(
    bin_path: str | Path,
    inputs: dict[str, np.ndarray],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    graph_index: int | None = None,
    num_graphs: int = 1,
) -> dict[str, np.ndarray]:
    """Run a precompiled QNN context binary (.bin, loaded via --retrieve_context).

    Some context binaries (e.g. this repo's Qwen3-1.7B split-stage artifacts)
    bundle more than one graph -- typically a prompt/prefill graph and a
    decode graph sharing one set of weights. Pass ``graph_index`` (0-based,
    matching the graph order reported by ``qnn-context-binary-utility.exe``)
    and ``num_graphs`` (the total graph count in the binary) to select one;
    qnn-net-run.exe requires every other graph slot to be explicitly skipped
    (``__``) rather than just omitted.
    """
    return _run(
        model_flag=["--retrieve_context", str(bin_path)],
        inputs=inputs,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend=backend,
        timeout_sec=timeout_sec,
        graph_index=graph_index,
        num_graphs=num_graphs,
    )


def run_context_binary_batch(
    bin_path: str | Path,
    inputs_list: list[dict[str, np.ndarray]],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    graph_index: int | None = None,
    num_graphs: int = 1,
) -> list[dict[str, np.ndarray]]:
    """Batched sibling of run_context_binary() -- see run_dlc_batch()."""
    return _run_batch(
        model_flag=["--retrieve_context", str(bin_path)],
        inputs_list=inputs_list,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend=backend,
        timeout_sec=timeout_sec,
        graph_index=graph_index,
        num_graphs=num_graphs,
    )


def _run(
    model_flag: list[str],
    inputs: dict[str, np.ndarray],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    graph_index: int | None = None,
    num_graphs: int = 1,
) -> dict[str, np.ndarray]:
    return _run_batch(
        model_flag=model_flag,
        inputs_list=[inputs],
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend=backend,
        timeout_sec=timeout_sec,
        graph_index=graph_index,
        num_graphs=num_graphs,
    )[0]


def _run_batch(
    model_flag: list[str],
    inputs_list: list[dict[str, np.ndarray]],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    graph_index: int | None = None,
    num_graphs: int = 1,
) -> list[dict[str, np.ndarray]]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _run_batch_once(
                model_flag,
                inputs_list,
                output_names,
                output_shapes,
                output_dtypes,
                backend,
                timeout_sec,
                graph_index,
                num_graphs,
            )
        except RuntimeError as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS:
                continue
    raise RuntimeError(
        f"qnn-net-run failed after {MAX_ATTEMPTS} attempts (model={model_flag[-1]}). "
        f"Last error:\n{last_exc}"
    ) from last_exc


def _run_batch_once(
    model_flag: list[str],
    inputs_list: list[dict[str, np.ndarray]],
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    output_dtypes: dict[str, np.dtype],
    backend: str,
    timeout_sec: float,
    graph_index: int | None = None,
    num_graphs: int = 1,
) -> list[dict[str, np.ndarray]]:
    global _profile_call_count

    if backend not in BACKENDS:
        raise ValueError(
            f"unsupported QNN backend {backend!r}; expected one of {sorted(BACKENDS)}"
        )
    if not QNN_NET_RUN.exists():
        raise RuntimeError(
            f"qnn-net-run executable not found at {QNN_NET_RUN}; set QAIRT_ROOT or QNN_NET_RUN"
        )
    if not model_flag[-1] or not Path(model_flag[-1]).exists():
        raise FileNotFoundError(f"QNN model artifact not found: {model_flag[-1]}")

    dump_layers = DEBUG_DIR is not None and model_flag[0] == "--dlc_path"

    work_dir = _SCRATCH_ROOT / f"run_{uuid.uuid4().hex[:12]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        list_lines = []
        for idx, item in enumerate(inputs_list):
            line_parts = []
            for name, array in item.items():
                raw_name = f"{name}_{idx}.raw"
                _write_raw(work_dir / raw_name, array)
                line_parts.append(f"{name}:={raw_name}")
            list_lines.append(" ".join(line_parts))
        input_list_path = work_dir / "input_list.txt"
        input_list_path.write_text("\n".join(list_lines) + "\n")

        if graph_index is not None:
            if not (0 <= graph_index < num_graphs):
                raise ValueError(
                    f"graph_index {graph_index} out of range for num_graphs={num_graphs}"
                )
            # qnn-net-run.exe requires one input-list slot per graph in the
            # binary when it contains more than one; "__" skips a graph.
            slots = ["__"] * num_graphs
            slots[graph_index] = input_list_path.name
            input_list_arg = ",".join(slots)
        else:
            input_list_arg = input_list_path.name

        output_dir = work_dir / "output"
        backend_dll = BACKENDS[backend]
        cmd = [
            str(QNN_NET_RUN),
            *model_flag,
            "--input_list",
            input_list_arg,
            "--backend",
            str(backend_dll),
            "--output_dir",
            str(output_dir.name),
            "--use_native_input_files",
            "--use_native_output_files",
        ]
        if PROFILE_DIR is not None:
            cmd += ["--profiling_level", _profile_level]
        if dump_layers:
            cmd += ["--debug"]
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                env=_env(),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"qnn-net-run timed out after {timeout_sec}s (model={model_flag[-1]}, "
                f"batch_size={len(inputs_list)}). Real inference normally finishes in low "
                f"tens of seconds even on CPU backend, so a hang this long usually means "
                f"another process is still holding the NPU/DSP session open (check for "
                f"orphaned qnn-net-run/geniex processes with Get-Process and kill them) "
                f"rather than genuine slowness.\n"
                f"partial stdout:\n{exc.stdout}\npartial stderr:\n{exc.stderr}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"qnn-net-run failed (exit {result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        if PROFILE_DIR is not None:
            _save_profile(model_flag, output_dir)

        outputs_list: list[dict[str, np.ndarray]] = []
        for idx in range(len(inputs_list)):
            result_dir = output_dir / f"Result_{idx}"
            if not result_dir.is_dir():
                # Multi-graph binaries nest results one level deeper, under
                # the selected graph's name (output_dir/<graph_name>/Result_N).
                nested = list(output_dir.glob(f"*/Result_{idx}"))
                if len(nested) == 1:
                    result_dir = nested[0]
                elif len(nested) > 1:
                    raise RuntimeError(
                        f"Ambiguous Result_{idx} directory across graphs: {nested}. "
                        f"Pass graph_index to select exactly one graph."
                    )

            if dump_layers:
                _save_layer_dump(model_flag, result_dir, item_index=idx)

            outputs: dict[str, np.ndarray] = {}
            for name in output_names:
                raw_file = result_dir / f"{name}_native.raw"
                if not raw_file.exists():
                    raw_file = result_dir / f"{name}.raw"
                if not raw_file.exists():
                    raise FileNotFoundError(
                        f"Expected output tensor '{name}' not found at {raw_file}. "
                        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                    )
                arr = np.fromfile(raw_file, dtype=output_dtypes[name])
                outputs[name] = arr.reshape(output_shapes[name])
            outputs_list.append(outputs)
        return outputs_list
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
