"""End-to-end voice Q&A pipeline: record a question -> transcribe it (Whisper)
-> get an answer from the local LLM (Qwen3-4B via Genie) -> speak the answer
back (MeloTTS). If Genie decides it doesn't have enough information to answer,
it asks a follow-up question out loud instead of guessing; the user records a
reply and the loop continues (bounded by --max-clarifications) until there's a
real answer.

Genie is a plain text-completion CLI (no schema-constrained decoding), so it's
asked to always reply with a fenced ```json block containing exactly
{"status": "answer"|"clarify", "text": "..."}. Clarification turns carry prior
turns as multi-turn ChatML history (see _build_multiturn_prompt), trimmed to
fit comfortably under Genie's 4096-token total context budget (see
_trim_history) -- unlike doctor_note_pipeline.py's map-reduce Genie calls,
which are each single-shot with no cross-call memory.
"""

from __future__ import annotations

import json
import re
import sys
import winsound
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import chunking  # noqa: E402
import onnxrt_runner  # noqa: E402
import patient_records  # noqa: E402
import record_audio  # noqa: E402
from doctor_note_pipeline import _synthesize_chunk_with_retry  # noqa: E402
from genie_runner import run_genie  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. A user's spoken question has been "
    "transcribed for you; your reply will be converted back to speech, so "
    "write plainly, as you would say it out loud. "
    "If the question is genuinely ambiguous or missing information you need "
    "to answer well, ask ONE short clarifying question instead of guessing. "
    "Respond with ONLY a fenced ```json ``` code block containing a JSON "
    "object with exactly two keys: 'status' (either \"answer\" or "
    "\"clarify\") and 'text' (the answer, or the clarifying question), no "
    "other text."
)

# Appended to SYSTEM_PROMPT only when --records-dir is set, so behavior with
# no records dir stays byte-for-byte identical to before this feature. Adds a
# third status value distinct from "clarify": "clarify" is for asking the
# *user* something their chart wouldn't contain, "need_records" is for
# information that might be in the patient's historical visit records but
# isn't in front of the model yet.
_RECORDS_INSTRUCTIONS = (
    "\n\nYou also have access to this patient's historical visit records "
    "below. A third 'status' value is now allowed: \"need_records\" -- use "
    "it when the answer might be in the patient's historical visit records "
    "but the detail you have so far (visit summaries only) isn't enough. "
    "The system will automatically fetch OCR'd scans of doctor's notes and "
    "AVS documents for you and ask again -- do not ask the user for this "
    "yourself. Use \"clarify\" only for things the patient's records could "
    "not contain. When using \"need_records\", still respond with only the "
    "same fenced json block, using exactly the 'status' and 'text' keys "
    "('text' can restate what you're trying to find)."
)

DEFAULT_MAX_CLARIFICATIONS = 5
DEFAULT_MAX_HISTORY_TOKENS = 3200  # leaves headroom under Genie's 4096 total budget for the system prompt + generation

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*)", re.DOTALL)


def _find_balanced_json_object(text: str) -> str | None:
    """Same brace-balanced scan as doctor_note_pipeline.py's helper of the
    same name -- duplicated rather than imported, matching this repo's
    existing convention of small self-contained helpers per pipeline script."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_block(text: str) -> dict:
    """Pulls {"status", "text"} out of Genie's response, tolerating a missing
    closing fence exactly like doctor_note_pipeline._extract_json_block. Fails
    open on a malformed/unparseable reply -- rather than crashing the whole
    voice loop over one bad Genie turn, treats the raw text as a direct answer
    so the user still gets *something* spoken back."""
    match = _JSON_BLOCK_RE.search(text)
    candidate = _find_balanced_json_object(match.group(1) if match else text)
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            if parsed.get("status") in ("answer", "clarify", "need_records") and "text" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    print(f"  [warning] could not parse a status/text JSON object out of Genie's reply; treating it as a direct answer:\n{text}")
    return {"status": "answer", "text": text.strip()}


def _current_system_prompt(records_context: str | None, avs_context: str | None) -> str:
    """Assembles the system prompt Genie actually sees this turn: the base
    prompt, plus (only when --records-dir is set) the records-fallback
    instructions and whatever patient-record context has been loaded so far.
    Used both to build the real prompt and, via _trim_history, to measure it,
    so token-budget trimming never diverges from what's actually sent."""
    if records_context is None:
        return SYSTEM_PROMPT
    parts = [SYSTEM_PROMPT, _RECORDS_INSTRUCTIONS, f"\n\nPatient visit summaries:\n{records_context}"]
    if avs_context is not None:
        parts.append(f"\n\nOCR'd doctor's notes / AVS scans:\n{avs_context}")
    return "".join(parts)


def _build_multiturn_prompt(system: str, history: list[tuple[str, str]]) -> str:
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for role, content in history:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _trim_history(
    history: list[tuple[str, str]], system: str, max_tokens: int = DEFAULT_MAX_HISTORY_TOKENS
) -> list[tuple[str, str]]:
    """Drops the oldest (user, assistant) turn-pair at a time until the
    assembled prompt's real Genie token count fits under max_tokens, so a long
    clarification loop can't silently overflow the 4096-token total budget."""
    trimmed = list(history)
    while len(trimmed) > 2 and chunking.count_genie_tokens(_build_multiturn_prompt(system, trimmed)) > max_tokens:
        trimmed = trimmed[2:]
    return trimmed


def synthesize_and_play(
    text: str, output_wav_path: str, backend: str, scratch_dir: Path, play: bool
) -> str:
    text_chunks = chunking.chunk_text_for_tts(text)
    audio_pieces = []
    sample_rate = None
    for i, chunk in enumerate(text_chunks):
        for audio, sr in _synthesize_chunk_with_retry(chunk, scratch_dir, f"{Path(output_wav_path).stem}_{i:03d}", backend):
            sample_rate = sample_rate or sr
            audio_pieces.append(audio)

    combined = np.concatenate(audio_pieces)
    sf.write(output_wav_path, combined, samplerate=sample_rate)
    if play:
        winsound.PlaySound(output_wav_path, winsound.SND_FILENAME)
    return output_wav_path


def run(
    input_wav: str | None = None,
    output_dir: str | None = None,
    backend: str = "htp",
    whisper: str = "base",
    play: bool = True,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
    records_dir: str | None = None,
    ocr_runtime: str = "dlc",
) -> dict:
    if whisper == "base":
        import whisper_onnx_pipeline as whisper_module
    else:
        import whisper_pipeline as whisper_module

    out_dir = Path(output_dir) if output_dir else Path(input_wav).parent if input_wav else Path("scratch/voice_qa")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_context = None
    avs_context = None
    if records_dir:
        records_context, dropped = patient_records.load_summaries_text(records_dir)
        if dropped:
            print(f"  [warning] visit summaries dropped (context budget): {dropped}")

    audio_path = input_wav or record_audio.record_to_wav(str(out_dir / "question.wav"))
    history: list[tuple[str, str]] = []
    clarify_count = 0

    while True:
        print(f"[transcribing] {audio_path}")
        transcript = whisper_module.run(audio_path, backend=backend)
        print(f"  Transcript: {transcript}")
        history.append(("user", transcript))

        if whisper == "base":
            # whisper_onnx_pipeline keeps its NPU session open in a persistent
            # worker for fast repeat calls; Genie/MeloTTS need exclusive HTP
            # access next, so release it now rather than fail with err 5000.
            onnxrt_runner.shutdown_worker()

        # Inner loop: same transcript, re-asked with more patient-record
        # context each time Genie escalates via "need_records" -- this never
        # consumes a clarification round-trip or records new audio.
        while True:
            print("[asking Genie]")
            system = _current_system_prompt(records_context, avs_context)
            prompt = _build_multiturn_prompt(system, _trim_history(history, system, max_history_tokens))
            raw = run_genie(prompt, backend=backend)
            parsed = _extract_json_block(raw)
            history.append(("assistant", raw))

            if parsed["status"] == "need_records" and records_dir and avs_context is None:
                print("  Genie needs patient record detail beyond the summaries; OCR'ing doctor's notes/AVS scans...")
                avs_context, dropped = patient_records.ocr_visit_images_text(records_dir, backend=backend, runtime=ocr_runtime)
                if dropped:
                    print(f"  [warning] OCR'd visit records dropped (context budget): {dropped}")
                continue

            if parsed["status"] == "need_records":
                # Already fetched everything available for this patient --
                # nothing left to escalate to, so ask the user instead of
                # looping forever.
                parsed["status"] = "clarify"
            break

        if parsed["status"] == "answer":
            final_text = parsed["text"]
            break

        if clarify_count >= max_clarifications:
            print(f"  Hit --max-clarifications={max_clarifications}; forcing a best-effort answer.")
            force_system = (
                _current_system_prompt(records_context, avs_context)
                + " You must respond with status \"answer\" now -- no more "
                "clarifying questions are allowed. Give your best-effort "
                "answer given everything discussed so far, even if some "
                "ambiguity remains."
            )
            force_prompt = _build_multiturn_prompt(force_system, _trim_history(history, force_system, max_history_tokens))
            final_text = _extract_json_block(run_genie(force_prompt, backend=backend))["text"]
            break

        clarify_count += 1
        print(f"  Genie needs clarification ({clarify_count}/{max_clarifications}): {parsed['text']}")
        clarify_wav = out_dir / f"clarify_{clarify_count}.wav"
        synthesize_and_play(parsed["text"], str(clarify_wav), backend, out_dir, play)
        audio_path = record_audio.record_to_wav(str(out_dir / f"reply_{clarify_count}.wav"))

    print(f"[answer] {final_text}")
    answer_wav = out_dir / "answer.wav"
    synthesize_and_play(final_text, str(answer_wav), backend, out_dir, play)

    history_path = out_dir / "conversation.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "answer_text": final_text,
        "answer_wav": str(answer_wav),
        "history_path": str(history_path),
        "clarifications": clarify_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="WAV file to use as the first question instead of recording live.")
    parser.add_argument("--backend", choices=["cpu", "htp"], default="htp")
    parser.add_argument("--whisper", choices=["base", "tiny"], default="base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--max-clarifications", type=int, default=DEFAULT_MAX_CLARIFICATIONS)
    parser.add_argument("--max-history-tokens", type=int, default=DEFAULT_MAX_HISTORY_TOKENS)
    parser.add_argument("--records-dir", default=None, help="Patient records folder (one subfolder per visit; see patient_records.py) to fall back on for questions the live conversation can't answer.")
    parser.add_argument("--ocr-runtime", choices=["dlc", "onnx"], default="dlc")
    args = parser.parse_args()

    result = run(
        input_wav=args.input,
        output_dir=args.output_dir,
        backend=args.backend,
        whisper=args.whisper,
        play=not args.no_play,
        max_clarifications=args.max_clarifications,
        max_history_tokens=args.max_history_tokens,
        records_dir=args.records_dir,
        ocr_runtime=args.ocr_runtime,
    )
    print(json.dumps(result, indent=2))
