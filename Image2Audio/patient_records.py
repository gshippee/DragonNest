"""Patient visit-history lookup for voice_qa_pipeline.py's records fallback.

Folder convention (new -- no real records folder existed anywhere in this
repo prior to this module, so this defines it): a single patient's
--records-dir contains one subfolder per previous visit, natural-sorted by
folder name (e.g. "visit_2" before "visit_10"), each holding that visit's
summary as plain text (any *.txt) and photos of that visit's doctor's notes
and AVS (any other image file -- both need OCR the same way, so this module
doesn't try to tell them apart).

Reuses doctor_note_pipeline.py's image discovery/OCR/formatting wholesale
(a visit folder's images are exactly the "directory of page images" shape
that pipeline already handles) rather than duplicating it -- matching this
repo's existing convention of importing another pipeline's private helpers
(e.g. voice_qa_pipeline.py importing _synthesize_chunk_with_retry).

Both load_summaries_text and ocr_visit_images_text pack visits most-recent-
first under a Genie token budget (chunking.count_genie_tokens), since
Genie's 4096-token total context must also fit the system prompt, live
conversation history, and generation -- so older visits are the ones that
get dropped if everything doesn't fit. Callers get told which visits were
dropped (returned, not swallowed) so a run() can log it instead of silently
losing history.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import chunking  # noqa: E402
from doctor_note_pipeline import (  # noqa: E402
    _natural_sort_key,
    extract_ocr_pages,
    format_ocr_pages,
)

RECORDS_CONTEXT_MAX_TOKENS = 2000

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def discover_visits(records_dir: str | Path) -> list[Path]:
    """Natural-sorted visit subfolders of records_dir (e.g. "visit_2" before
    "visit_10", matching doctor_note_pipeline's multi-page ordering)."""
    records_dir = Path(records_dir)
    visits = [p for p in records_dir.iterdir() if p.is_dir()]
    if not visits:
        raise ValueError(f"No visit subfolders found in records dir: {records_dir}")
    return sorted(visits, key=_natural_sort_key)


def _pack_most_recent_first(
    visits: list[Path], render: "callable[[Path], str | None]", max_tokens: int
) -> tuple[str, list[str]]:
    """Shared packing logic for load_summaries_text/ocr_visit_images_text:
    render() each visit (newest first), keep adding whole visits while the
    assembled text's real Genie token count stays under max_tokens, then
    re-emit the kept visits in chronological order for readability. A visit
    whose render() returns None/empty (e.g. no *.txt file in that folder) is
    skipped, not counted as dropped-for-budget."""
    kept: list[tuple[Path, str]] = []
    dropped: list[str] = []
    running_tokens = 0

    for visit in reversed(visits):
        text = render(visit)
        if not text:
            continue
        block = f"--- Visit: {visit.name} ---\n{text}"
        block_tokens = chunking.count_genie_tokens(block)
        if kept and running_tokens + block_tokens > max_tokens:
            dropped.append(visit.name)
            continue
        kept.append((visit, block))
        running_tokens += block_tokens

    kept.reverse()
    assembled = "\n\n".join(block for _visit, block in kept)
    dropped.reverse()
    return assembled, dropped


def load_summaries_text(
    records_dir: str | Path, max_tokens: int = RECORDS_CONTEXT_MAX_TOKENS
) -> tuple[str, list[str]]:
    """Read every visit's *.txt summary, packed most-recent-first under
    max_tokens. Returns (assembled_text, dropped_visit_names)."""
    visits = discover_visits(records_dir)

    def _render(visit: Path) -> str | None:
        txt_paths = sorted(visit.glob("*.txt"), key=_natural_sort_key)
        if not txt_paths:
            return None
        return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in txt_paths)

    return _pack_most_recent_first(visits, _render, max_tokens)


def ocr_visit_images_text(
    records_dir: str | Path,
    backend: str = "htp",
    runtime: str = "dlc",
    max_tokens: int = RECORDS_CONTEXT_MAX_TOKENS,
) -> tuple[str, list[str]]:
    """OCR every image in every visit folder (doctor's notes + AVS alike),
    packed most-recent-first under max_tokens. Expensive -- one EasyOCR pass
    per image -- so callers should compute this once per pipeline run and
    cache the result. Returns (assembled_text, dropped_visit_names)."""
    visits = discover_visits(records_dir)

    def _render(visit: Path) -> str | None:
        image_paths = [p for p in visit.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]
        if not image_paths:
            return None
        image_paths = sorted(image_paths, key=_natural_sort_key)
        pages = extract_ocr_pages(image_paths, backend=backend, runtime=runtime)
        return format_ocr_pages(pages)

    return _pack_most_recent_first(visits, _render, max_tokens)
