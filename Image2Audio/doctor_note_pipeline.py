"""End-to-end pipeline: a photo of a typed doctor's note (or a directory of
multiple page photos of one visit's document) -> OCR (text, confidence, box)
lines per page, grouped into table rows by box y-overlap -> a human review
checkpoint -> a plain-language summary and next-steps drafted by Genie ->
speech audio, auto-played on completion.

Every model in this chain has a real input-size ceiling (see chunking.py's
docstring for specifics: EasyOCR's detector resolution, Genie's Qwen3-4B
context budget, MeloTTS's phone/BERT-token truncation) that this pipeline
respects by tiling/chunking inputs and running multiple passes rather than
one call on a potentially oversized input:
  - OCR runs once per page image, and once per tile within a large page
    image (chunking.tile_image_if_large).
  - Summarization runs two Genie passes per token-budgeted text chunk: an
    extraction pass that numbers each OCR row and asks Genie to reply with
    just the row numbers containing patient-specific medical facts (problems,
    allergies, medications, vitals, labs, follow-ups) -- classifying by index
    rather than asking the model to reproduce row text avoids both copy-
    fidelity drift and the multi-minute repetition loops a 4B model can fall
    into when asked to generate long verbatim output -- then the existing
    summarize-and-reconcile map-reduce over the (usually much shorter) kept
    rows. This both lets a multi-page document exceed Qwen3's 4096-token
    context and keeps boilerplate-heavy pages (patient-satisfaction surveys,
    smoking-cessation app listings, etc.) from crowding out real medical
    content in the final summary.
  - TTS runs once per phone-budgeted text chunk (chunking.chunk_text_for_tts),
    and the resulting audio chunks are concatenated into one final .wav.

Genie is asked for plain text (a SUMMARY:/NEXT STEPS: block), not JSON --
genie-t2t-run.exe has no schema-constrained decoding, so asking it to emit
JSON and then brace-matching the response out of raw text was the most
fragile part of this pipeline. The pipeline pauses after OCR for a human to
confirm or correct the extracted text (doctor's notes carry real risk if OCR
misreads e.g. a drug dosage) before continuing.
"""

from __future__ import annotations

import re
import sys
import winsound
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import chunking  # noqa: E402
import easyocr_pipeline  # noqa: E402
import melotts_pipeline  # noqa: E402
from genie_runner import build_chatml_prompt, run_genie  # noqa: E402

SYSTEM_PROMPT = "You are a careful medical assistant. Follow instructions exactly."

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_DIGIT_RUN_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part
        for part in _DIGIT_RUN_RE.split(path.stem)
    )


def _discover_image_paths(input_path: Path) -> list[Path]:
    """Resolve `input_path` into a natural-sorted list of page images: the
    path itself if it's a file, or every image file directly inside it
    (case-insensitive .png/.jpg/.jpeg) if it's a directory -- natural-sorted
    so page "2.png" sorts before "10.png" (plain string sort would not)."""
    if input_path.is_file():
        return [input_path]

    paths = [p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f"No image files (.png/.jpg/.jpeg) found in directory: {input_path}")
    return sorted(paths, key=_natural_sort_key)


# --- Stage 1: OCR (tiled) --------------------------------------------------


def extract_ocr_lines(
    image_path: str, backend: str = "htp", runtime: str = "dlc", max_image_dim: int = chunking.DEFAULT_MAX_IMAGE_DIM
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    image = Image.open(image_path).convert("RGB")
    tiles = chunking.tile_image_if_large(image, max_dim=max_image_dim)

    app = easyocr_pipeline.build_app(backend=backend, runtime=runtime)
    blocks = []
    for tile_image, (x_offset, y_offset) in tiles:
        tile_lines = []
        for _annotated, texts, confidences, boxes in app.predict_text_from_image(tile_image):
            for text, confidence, (xmin, xmax, ymin, ymax) in zip(texts, confidences, boxes, strict=False):
                # Shift each tile's local box into original-image coordinates
                # so row grouping (chunking.group_lines_into_rows) works
                # correctly across tile boundaries.
                tile_lines.append(
                    (text, confidence, (xmin + x_offset, xmax + x_offset, ymin + y_offset, ymax + y_offset))
                )
        blocks.append(tile_lines)
        if len(tiles) > 1:
            print(f"  OCR'd tile at offset ({x_offset}, {y_offset}): {len(tile_lines)} text box(es)")

    return chunking.dedupe_ocr_lines(blocks)


def extract_ocr_pages(
    image_paths: list[Path], backend: str = "htp", runtime: str = "dlc", max_image_dim: int = chunking.DEFAULT_MAX_IMAGE_DIM
) -> list[list[tuple[str, float, tuple[int, int, int, int]]]]:
    """Run extract_ocr_lines once per image path, so each page's OCR lines
    keep their own image's box coordinate space (row-grouping across pages
    would be meaningless -- two different images' pixel coordinates have no
    relationship to each other)."""
    pages = []
    for i, image_path in enumerate(image_paths, start=1):
        lines = extract_ocr_lines(str(image_path), backend=backend, runtime=runtime, max_image_dim=max_image_dim)
        print(f"  OCR'd page {i}/{len(image_paths)}: {image_path.name} ({len(lines)} text box(es))")
        pages.append(lines)
    return pages


def format_ocr_lines(lines: list[tuple[str, float | None, tuple[int, int, int, int] | None]]) -> str:
    """Render OCR lines as one row per line, with columns of the same row
    joined by ' | ' (chunking.group_lines_into_rows reconstructs rows from
    box y-overlap, so a table's columns land on one line together instead of
    scattering across separate disconnected lines). Each cell is prefixed
    with its confidence (0.00-1.00), or 'N/A' for human-corrected text, which
    has no box and so isn't grouped -- see group_lines_into_rows."""
    rows = chunking.group_lines_into_rows(lines)
    return "\n".join(
        " | ".join(f"{'N/A' if c is None else f'{c:.2f}'}\t{t}" for t, c, _b in row)
        for row in rows
    )


def format_ocr_pages(pages: list[list[tuple[str, float | None, tuple[int, int, int, int] | None]]]) -> str:
    """Render a document's OCR pages for the review printout / Genie prompt.
    A single page renders exactly like format_ocr_lines (so single-image runs
    are unaffected byte-for-byte by multi-page support). With more than one
    page, each page's rows are joined with a blank line between them instead
    of format_ocr_lines' single newline -- table rows rarely end in
    sentence-ending punctuation, so chunking.chunk_text_for_genie's
    paragraph-then-sentence splitter needs that blank line to find a
    row-granularity unit to pack once a multi-page transcript is long enough
    to need map-reduce. Page headers are plain text spliced between pages,
    never fed through group_lines_into_rows, so they can't trigger its
    all-or-nothing box=None grouping behavior."""
    if len(pages) == 1:
        return format_ocr_lines(pages[0])

    n = len(pages)
    page_texts = []
    for i, lines in enumerate(pages, start=1):
        rows = chunking.group_lines_into_rows(lines)
        row_texts = [
            " | ".join(f"{'N/A' if c is None else f'{c:.2f}'}\t{t}" for t, c, _b in row)
            for row in rows
        ]
        page_texts.append(f"--- Page {i} of {n} ---\n\n" + "\n\n".join(row_texts))
    return "\n\n".join(page_texts)


# --- Stage 2: human review checkpoint --------------------------------------


def review_ocr_lines(
    lines: list[tuple[str, float, tuple[int, int, int, int] | None]], auto_accept: bool = False
) -> list[tuple[str, float | None, tuple[int, int, int, int] | None]]:
    print("\n--- OCR text (confidence<TAB>text, review before continuing) ---")
    print(format_ocr_lines(lines))
    print("--- end OCR text ---\n")
    if auto_accept:
        print("(--auto-accept-ocr set: skipping interactive review)")
        return lines
    choice = input("Press Enter to accept, or type 'edit' to provide corrected text: ").strip().lower()
    if choice != "edit":
        return lines

    print("Enter corrected text. Finish with a line containing only 'END'.")
    corrected = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        corrected.append((line, None, None))
    return corrected


def review_ocr_pages(
    pages: list[list[tuple[str, float, tuple[int, int, int, int] | None]]], auto_accept: bool = False
) -> list[list[tuple[str, float | None, tuple[int, int, int, int] | None]]]:
    """Multi-page counterpart to review_ocr_lines. A human correcting a
    multi-page document retypes one combined blob (same as the single-image
    edit flow), which has no box coordinates and so can't be re-split back
    into per-page groups -- it becomes a single page, same
    "manual correction loses box/page structure" degrade review_ocr_lines
    already accepts for single images."""
    print("\n--- OCR text (confidence<TAB>text, review before continuing) ---")
    print(format_ocr_pages(pages))
    print("--- end OCR text ---\n")
    if auto_accept:
        print("(--auto-accept-ocr set: skipping interactive review)")
        return pages
    choice = input("Press Enter to accept, or type 'edit' to provide corrected text: ").strip().lower()
    if choice != "edit":
        return pages

    print("Enter corrected text. Finish with a line containing only 'END'.")
    corrected = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        corrected.append((line, None, None))
    return [corrected]


# --- Stage 3: summarization (map-reduce over Genie) ------------------------


_SUMMARY_RE = re.compile(
    r"^SUMMARY:\s*(.*?)\s*^NEXT\s+STE+PS?:\s*(.*?)(?=^SUMMARY:|^NEXT\s+STE+PS?:|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
_MARKDOWN_STRIP_RE = re.compile(r"[*#`]")
_ANGLE_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


_NUMBER_RE = re.compile(r"\d+")
_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_EXTRACTION_BATCH_SIZE = 15


def _number_rows_for_extraction(chunk: str, batch_size: int = _EXTRACTION_BATCH_SIZE) -> list[tuple[str, list[str]]]:
    """Split a formatted OCR chunk into individual row-lines, number them, and
    group them into batches of at most `batch_size` rows -- one extraction
    prompt per batch, each numbered locally from 1. Asking a 4B model to
    classify by index (rather than reproduce row text) avoids copy-fidelity
    drift and repetition loops, but on a full ~50-row chunk it still
    unreliably enumerates all the way to the end before trailing off into
    rambling -- observed directly on Sample6, where a 56-row chunk's reply
    correctly listed rows up to ~37 and then degraded, silently dropping
    every row after that point (including the patient's actual medications).
    Keeping each batch short enough that a full, correct enumeration fits on
    one line makes that failure both less likely and far less costly on the
    rows it does affect. '--- Page N of M ---' headers are kept in place
    inline for context but never numbered or counted toward batch_size."""
    parsed: list[tuple[str, str]] = []
    for line in chunk.split("\n"):
        if not line.strip():
            continue
        if line.startswith("--- Page"):
            parsed.append(("header", line))
        else:
            parsed.append(("row", line))

    batches: list[tuple[str, list[str]]] = []
    rows: list[str] = []
    numbered_lines: list[str] = []
    for kind, text in parsed:
        if kind == "header":
            numbered_lines.append(text)
            continue
        rows.append(text)
        numbered_lines.append(f"[{len(rows)}] {text}")
        if len(rows) >= batch_size:
            batches.append(("\n".join(numbered_lines), rows))
            rows, numbered_lines = [], []
    if rows:
        batches.append(("\n".join(numbered_lines), rows))
    return batches


def _extract_facts_prompt(numbered_text: str, n: int) -> str:
    return (
        "The following is a numbered list of rows OCR'd from a photo of a "
        "page of a doctor's note or after-visit summary. Each row's table "
        "columns (e.g. a medication's dosage/frequency/instructions) are "
        "already joined together on that row separated by ' | ' -- read "
        "each row as one record, not as unrelated fragments. If the "
        "transcript contains '--- Page N of M ---' headers, treat pages as "
        "parts of one visit's document; those headers are not rows and have "
        "no number.\n\n"
        "Identify every row that is generic boilerplate, NOT specific to "
        "this patient or this visit: marketing text, satisfaction surveys, "
        "hotline/app/website directories, legal disclaimers, and general "
        "health education content that doesn't mention this patient's own "
        "facts. Keep everything else -- when in doubt, do NOT mark a row as "
        "boilerplate; only mark rows you are confident are generic. Never "
        "mark a row that contains patient name/identifiers, diagnoses/"
        "problems, allergies and reactions, medications with dosage/"
        "frequency/instructions, vitals, lab/imaging results and their "
        "values, or follow-up appointments/instructions.\n\n"
        f"Reply with ONLY the row numbers to remove as boilerplate, out of "
        f"1-{n}, comma-separated (e.g. 2,5,9). Do not reproduce any row's "
        "text. Do not add any other words, explanation, or punctuation. If "
        "no row is boilerplate, reply with exactly: NONE\n\n"
        f"{numbered_text}"
    )


def _parse_removed_row_numbers(response: str, n: int) -> set[int] | None:
    """Pull row numbers out of an extraction reply. Only the reply's first
    non-blank line is considered: even with no_think and salvage_partial, this
    4B model sometimes emits a clean, correct answer on its first line and
    then rambles on past it into self-doubting meta-commentary that itself
    contains stray digits (row numbers quoted while "reconsidering", fragments
    of a restated instruction, etc.) -- scanning the whole reply for digits
    would pick those up as if they were real answer content. The first line
    may use 'N-M' range shorthand (observed on Sample6 even though the prompt
    only asked for a comma-separated list), so ranges are expanded before
    plain digits are collected. Returns an empty set for an explicit,
    digit-free 'NONE' (nothing to remove -- keep the whole batch), or None if
    the first line couldn't be read as a number list at all (caller should
    fall back to keeping the whole batch too -- see summarize_note). Asking
    which rows to REMOVE as boilerplate, rather than which to KEEP, means any
    row the model fails to classify -- or that a parse/timeout failure drops
    entirely -- defaults to being kept rather than silently discarded, which
    matters because a prior "reply with rows to keep" framing was observed on
    Sample6 to drop real medications (desvenlafaxine/Pristiq, phentermine)
    that the model simply never got around to listing before rambling off
    topic."""
    text = response.strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    if first_line.upper().startswith("NONE") and not _NUMBER_RE.search(first_line):
        return set()
    found: set[int] = set()
    for lo, hi in _RANGE_RE.findall(first_line):
        lo, hi = int(lo), int(hi)
        if lo <= hi:
            found.update(i for i in range(lo, hi + 1) if 1 <= i <= n)
    ranged_text = _RANGE_RE.sub(" ", first_line)
    found.update(i for i in (int(m) for m in _NUMBER_RE.findall(ranged_text)) if 1 <= i <= n)
    return found if found else None


def _summarize_facts_prompt(facts_chunk: str) -> str:
    return (
        "The following lines are patient-specific facts already filtered "
        "out of a doctor's note or after-visit summary -- marketing text, "
        "surveys, and generic education content have already been removed, "
        "so treat every line here as relevant. Each line keeps its "
        "original OCR confidence prefix (0.00-1.00, or 'N/A' if a human "
        "reviewer corrected it); confidence is often lower for handwriting, "
        "drug names, and dosages -- treat text below about 0.50 with more "
        "skepticism and note where a value is ambiguous rather than "
        "confidently inventing a cleaner-looking one, but do not omit a "
        "fact just because it is low-confidence.\n\n"
        "First write a short plain-language summary of the visit (problem/"
        "diagnosis, allergies, medications, vitals, labs, symptoms) that "
        "includes as many of the concrete facts below as are relevant -- "
        "names, dosages, dates, and values matter, don't generalize them "
        "away. Copy medication names, dosages, and dates character-for-"
        "character exactly as written below -- never paraphrase, abbreviate, "
        "or 'correct' the spelling of a drug name, since misreading one can "
        "change which medication it refers to. Then suggest clear next steps "
        "for the patient based only on "
        "these facts. Respond in plain text only -- no markdown, no "
        "asterisks, no HTML tags, no other commentary. Write the summary as "
        "its own paragraph starting on the line right after the literal "
        "word 'SUMMARY:', then write the next steps as their own paragraph "
        "starting on the line right after the literal words 'NEXT STEPS:'.\n\n"
        f"Extracted facts:\n{facts_chunk}"
    )


def _dedupe_repeated_sentences(text: str) -> str:
    """Drop sentences that are a near-exact repeat of one already kept.
    Salvaged (timed-out) replies can loop on a single paragraph verbatim
    several times in a row before the subprocess is killed -- observed on
    Sample6, where a summarization pass repeated the same "text seems
    unclear due to inconsistencies" sentence ~6 times. Comparison is
    case-insensitive and whitespace-normalized so trivial reformatting
    doesn't defeat the dedupe; first occurrence of each sentence wins."""
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        key = _WHITESPACE_RE.sub(" ", sentence).lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
    return " ".join(kept)


def _parse_summary_response(text: str) -> tuple[str, str]:
    cleaned = _ANGLE_TAG_RE.sub("", _MARKDOWN_STRIP_RE.sub("", text))
    match = _SUMMARY_RE.search(cleaned)
    if not match:
        print("  (warning: model response didn't follow SUMMARY:/NEXT STEPS: format; using full response as summary)")
        return _dedupe_repeated_sentences(cleaned.strip()), ""
    return _dedupe_repeated_sentences(match.group(1).strip()), _dedupe_repeated_sentences(match.group(2).strip())


def _reconcile_summary_prompt(partials: list[tuple[str, str]]) -> str:
    joined = "\n\n".join(
        f"Chunk {i + 1} summary:\n{s}\nChunk {i + 1} next steps:\n{n}"
        for i, (s, n) in enumerate(partials)
    )
    return (
        "The following are summaries and next-steps drafted from different "
        "chunks of the same doctor's note. Merge them into one coherent "
        "summary and one coherent next-steps list, removing duplicates. Copy "
        "medication names, dosages, and dates character-for-character exactly "
        "as they already appear below -- never paraphrase, abbreviate, or "
        "'correct' a drug name's spelling while merging. "
        "Respond in plain text only -- no markdown, no asterisks, no HTML "
        "tags, no other commentary. Write the summary as its own paragraph "
        "starting on the line right after the literal word 'SUMMARY:', then "
        "write the next steps as their own paragraph starting on the line "
        "right after the literal words 'NEXT STEPS:'.\n\n"
        f"{joined}"
    )


def _summarize_with_retry(prompt: str, backend: str) -> tuple[str, str]:
    """Run one summarization/reconcile Genie call. If the reply doesn't
    follow the SUMMARY:/NEXT STEPS: format, retry once with a different
    sampler seed before falling back to using the raw reply as the summary --
    observed on Sample6, where the single chunk carrying the patient's name,
    allergies, and medications happened to be the one pass that rambled
    instead of following the format, so its facts were replaced by
    meta-commentary right before the reconcile step that most needed them.
    genie_config.json pins a fixed seed (42), so retrying with the same seed
    reproduces the identical malformed reply -- run_genie's seed override
    makes the retry an actually-independent sample instead of a no-op."""
    raw = run_genie(prompt, backend=backend, salvage_partial=True)
    cleaned = _ANGLE_TAG_RE.sub("", _MARKDOWN_STRIP_RE.sub("", raw))
    if _SUMMARY_RE.search(cleaned):
        return _parse_summary_response(raw)
    print("    (warning: reply didn't follow SUMMARY:/NEXT STEPS: format; retrying once)")
    raw = run_genie(prompt, backend=backend, salvage_partial=True, seed=1337)
    return _parse_summary_response(raw)


def summarize_note(
    pages: list[list[tuple[str, float | None, tuple[int, int, int, int] | None]]],
    backend: str = "htp",
    max_genie_tokens: int = chunking.DEFAULT_MAX_GENIE_TOKENS,
) -> tuple[str, str]:
    formatted = format_ocr_pages(pages)
    ocr_chunks = chunking.chunk_text_for_genie(formatted, max_tokens=max_genie_tokens)
    print(f"Extraction: {len(ocr_chunks)} OCR chunk(s) to scan for patient-specific facts.")

    fact_blocks = []
    for i, chunk in enumerate(ocr_chunks):
        batches = _number_rows_for_extraction(chunk)
        print(f"  Genie extraction pass {i + 1}/{len(ocr_chunks)} ({len(batches)} batch(es))...")
        for j, (numbered_text, rows) in enumerate(batches):
            if not rows:
                continue
            prompt = build_chatml_prompt(
                SYSTEM_PROMPT, _extract_facts_prompt(numbered_text, len(rows)), no_think=True
            )
            response = run_genie(prompt, backend=backend, salvage_partial=True)
            removed = _parse_removed_row_numbers(response, len(rows))
            if removed is None:
                print(f"    (warning: batch {j + 1}/{len(batches)}: couldn't parse row numbers; keeping full batch)")
                fact_blocks.append("\n".join(rows))
            else:
                kept_rows = [r for k, r in enumerate(rows, start=1) if k not in removed]
                if kept_rows:
                    fact_blocks.append("\n".join(kept_rows))

    if not fact_blocks:
        print("  (warning: no patient-specific facts extracted; falling back to raw OCR text)")
        fact_blocks = ocr_chunks

    facts_text = "\n\n".join(fact_blocks)
    chunks = chunking.chunk_text_for_genie(facts_text, max_tokens=max_genie_tokens)
    print(f"Summarization: {len(chunks)} chunk(s) of extracted facts.")

    partials = []
    for i, chunk in enumerate(chunks):
        print(f"  Genie summarization pass {i + 1}/{len(chunks)}...")
        prompt = build_chatml_prompt(SYSTEM_PROMPT, _summarize_facts_prompt(chunk), no_think=True)
        partials.append(_summarize_with_retry(prompt, backend))

    if len(partials) == 1:
        return partials[0]

    print("  Reconciling partial summaries...")
    prompt = build_chatml_prompt(SYSTEM_PROMPT, _reconcile_summary_prompt(partials), no_think=True)
    return _summarize_with_retry(prompt, backend)


# --- Stage 4: TTS (chunked + concatenated) ---------------------------------


def _split_chunk_in_half(text: str) -> list[str]:
    """Fallback splitter for _synthesize_chunk_with_retry: split on the
    nearest sentence boundary to the midpoint, or on whitespace if there's
    only one sentence, so a chunk that still overflows the encoder's
    predicted-duration ceiling (see melotts_pipeline.TTSTextTooLongError) can
    be retried as two smaller pieces instead of producing static."""
    sentences = chunking._split_sentences(text)
    if len(sentences) > 1:
        mid = len(sentences) // 2
        return [" ".join(sentences[:mid]), " ".join(sentences[mid:])]
    words = text.split(" ")
    mid = max(1, len(words) // 2)
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _synthesize_chunk_with_retry(
    chunk: str, scratch_dir: Path, chunk_id: str, backend: str
) -> list[tuple]:
    """Runs melotts_pipeline.run() on `chunk`; if the encoder's predicted
    duration overflows its fixed frame budget (TTSTextTooLongError -- can
    happen even under the phone-count budget, since punctuation/numerals can
    inflate predicted duration disproportionately to phone count), splits the
    chunk in half and retries each half recursively rather than emitting
    static. Returns a list of (audio, sample_rate) pieces in playback order."""
    chunk_wav = scratch_dir / f"_tts_chunk_{chunk_id}.wav"
    try:
        melotts_pipeline.run(chunk, str(chunk_wav), backend=backend)
    except melotts_pipeline.TTSTextTooLongError:
        if len(chunk.split(" ")) <= 1:
            raise
        print(f"  chunk {chunk_id} overflowed predicted duration; splitting and retrying...")
        pieces = []
        for i, half in enumerate(_split_chunk_in_half(chunk)):
            pieces.extend(_synthesize_chunk_with_retry(half, scratch_dir, f"{chunk_id}_{i}", backend))
        return pieces
    audio, sr = sf.read(chunk_wav)
    chunk_wav.unlink(missing_ok=True)
    return [(audio, sr)]


def synthesize_speech(
    text: str,
    output_wav_path: str,
    backend: str = "htp",
    max_tts_phones: int = chunking.DEFAULT_MAX_TTS_PHONES,
    scratch_dir: Path | None = None,
) -> str:
    text_chunks = chunking.chunk_text_for_tts(text, max_phones=max_tts_phones)
    print(f"TTS: {len(text_chunks)} chunk(s) of text.")

    scratch_dir = scratch_dir or Path(output_wav_path).parent
    scratch_dir.mkdir(parents=True, exist_ok=True)

    audio_pieces = []
    sample_rate = None
    for i, chunk in enumerate(text_chunks):
        print(f"  TTS pass {i + 1}/{len(text_chunks)}...")
        for audio, sr in _synthesize_chunk_with_retry(chunk, scratch_dir, f"{i:03d}", backend):
            sample_rate = sample_rate or sr
            audio_pieces.append(audio)

    combined = np.concatenate(audio_pieces)
    sf.write(output_wav_path, combined, samplerate=sample_rate)
    return output_wav_path


def play_wav(wav_path: str) -> None:
    winsound.PlaySound(wav_path, winsound.SND_FILENAME)


# --- Orchestration ----------------------------------------------------------


def run(
    image_path: str,
    output_dir: str | None = None,
    backend: str = "htp",
    runtime: str = "dlc",
    play: bool = True,
    max_image_dim: int = chunking.DEFAULT_MAX_IMAGE_DIM,
    max_genie_tokens: int = chunking.DEFAULT_MAX_GENIE_TOKENS,
    max_tts_phones: int = chunking.DEFAULT_MAX_TTS_PHONES,
    auto_accept_ocr: bool = False,
) -> dict:
    image_path = Path(image_path)
    out_dir = Path(output_dir) if output_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem if image_path.is_file() else image_path.name

    image_paths = _discover_image_paths(image_path)
    print(f"[1/4] Running OCR on {len(image_paths)} image(s)...")
    ocr_pages = extract_ocr_pages(image_paths, backend=backend, runtime=runtime, max_image_dim=max_image_dim)
    (out_dir / f"{stem}_ocr.txt").write_text(format_ocr_pages(ocr_pages), encoding="utf-8")

    print("[2/4] Awaiting review...")
    reviewed_pages = review_ocr_pages(ocr_pages, auto_accept=auto_accept_ocr)
    if reviewed_pages != ocr_pages:
        (out_dir / f"{stem}_ocr_corrected.txt").write_text(format_ocr_pages(reviewed_pages), encoding="utf-8")

    print("[3/4] Summarizing note...")
    summary, next_steps = summarize_note(reviewed_pages, backend=backend, max_genie_tokens=max_genie_tokens)
    summary_path = out_dir / f"{stem}_summary.txt"
    summary_path.write_text(f"Summary:\n{summary}\n\nNext steps:\n{next_steps}\n", encoding="utf-8")
    print(f"Summary:\n{summary}\n\nNext steps:\n{next_steps}")

    print("[4/4] Synthesizing speech...")
    wav_path = out_dir / f"{stem}_email.wav"
    synthesize_speech(
        f"Here is a summary of your visit. {summary} Next steps. {next_steps}",
        str(wav_path),
        backend=backend,
        max_tts_phones=max_tts_phones,
        scratch_dir=out_dir,
    )
    print(f"Audio saved to {wav_path}")

    if play:
        print("Playing audio...")
        play_wav(str(wav_path))

    return {
        "ocr_text_path": str(out_dir / f"{stem}_ocr.txt"),
        "summary_path": str(summary_path),
        "wav_path": str(wav_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        help="Path to a single doctor's-note image, or a directory of page images (e.g. a "
        "multi-page visit summary) that will be natural-sorted and summarized together.",
    )
    parser.add_argument("--backend", choices=["cpu", "htp"], default="htp")
    parser.add_argument("--runtime", choices=["dlc", "onnx"], default="dlc")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument(
        "--auto-accept-ocr",
        action="store_true",
        help="Skip the interactive OCR review checkpoint (for automated/non-interactive runs).",
    )
    parser.add_argument("--max-image-dim", type=int, default=chunking.DEFAULT_MAX_IMAGE_DIM)
    parser.add_argument("--max-genie-tokens", type=int, default=chunking.DEFAULT_MAX_GENIE_TOKENS)
    parser.add_argument("--max-tts-phones", type=int, default=chunking.DEFAULT_MAX_TTS_PHONES)
    args = parser.parse_args()

    run(
        args.image_path,
        output_dir=args.output_dir,
        backend=args.backend,
        runtime=args.runtime,
        play=not args.no_play,
        max_image_dim=args.max_image_dim,
        max_genie_tokens=args.max_genie_tokens,
        max_tts_phones=args.max_tts_phones,
        auto_accept_ocr=args.auto_accept_ocr,
    )
