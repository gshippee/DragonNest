"""Shared size-measurement/splitting utilities for doctor_note_pipeline.py.

Each of the three models it drives has a real, hard input ceiling that is
silently exceeded (truncated or degraded) rather than raising a clear error:
  - EasyOCR's detector letterboxes any image down to a fixed (608, 800) --
    a large/dense note downsized that far can make small text illegible
    before the detector ever sees it, so large images are tiled into
    overlapping crops and each crop is OCR'd separately.
  - Genie's Qwen3-4B has a fixed total context budget (prompt + generation
    tokens together, see genie_config.json's "size": 4096) -- long OCR text
    must be split into token-budgeted windows and processed as map-reduce.
  - MeloTTSApp.preprocess_text pads/*truncates* to MAX_SEQ_LEN=512 phones and
    MAX_BERT_TOKENS=200 BERT tokens with no error -- long email text must be
    split into chunks that individually stay under that budget, synthesized
    separately, and concatenated.

All three follow the same shape: measure real size -> split if over budget,
never mid-sentence -> caller processes each piece -> caller merges results.
"""

from __future__ import annotations

import re

from PIL import Image

import device_config

QWEN_DIR = str(device_config.GENIE_DIR)

# MeloTTS's real ceiling is MAX_SEQ_LEN=512 phones (qai_hub_models/models/
# _shared/melotts/model.py) -- MAX_BERT_TOKENS=200 is a separate model
# (t5_encoder/t5_decoder/bert_wrapper.bin) that melotts_pipeline.py's actual
# code path never calls, so it isn't a real constraint here; only the 512
# phone cap on the encoder/flow/decoder path matters.
#
# A chars-per-phone estimate was tried first but is not reliable enough:
# measured in practice on a real email chunk, 359 characters produced 669
# phones (~1.86 phones/char, nearly double a "typical" English estimate),
# which silently passed a char-count budget while blowing past the 512-phone
# cap. preprocess_text() truncates the phone tensor to MAX_SEQ_LEN but still
# returns the pre-truncation phone_len, so the encoder predicts audio length
# from the inflated count while flow.bin's output is capped at
# UPSAMPLED_MAX_SEQ_LEN -- the final audio gets trimmed to a length longer
# than what was actually synthesized, and the tail plays back the decoder's
# zero-padded buffer as constant static. Measuring phones exactly (below)
# costs one CPU-only G2P/BERT preprocessing call per candidate chunk, which
# is cheap relative to the actual NPU encoder/flow/decoder calls that follow.
DEFAULT_MAX_TTS_PHONES = 480  # safety margin below MAX_SEQ_LEN=512

_melo_tts_object = None


def _get_melo_tts_object():
    global _melo_tts_object
    if _melo_tts_object is None:
        from qai_hub_models.datasets.common_voice.voiceai_lang import TTSLanguage
        from qai_hub_models.models._shared.melotts.model import get_tts_object

        _melo_tts_object = get_tts_object(TTSLanguage.ENGLISH)
    return _melo_tts_object


def count_melo_phones(text: str) -> int:
    """Exact phone count for `text` via MeloTTS's own G2P preprocessing
    (the same call MeloTTSApp.preprocess_text makes), used instead of a
    chars-per-phone guess -- see module comment above for why the guess
    isn't safe."""
    from qai_hub_models.models._shared.melotts.app import get_text_for_tts_infer

    tts_object = _get_melo_tts_object()
    _bert, _ja_bert, phones, _tones, _lang_ids = get_text_for_tts_infer(
        text, tts_object.language, tts_object.hps, "cpu", tts_object.symbol_to_id
    )
    return phones.size(0)

DEFAULT_MAX_GENIE_TOKENS = 1500  # well under the 4096 total budget, leaving room for instructions + generation
DEFAULT_MAX_IMAGE_DIM = 1600  # px; EasyOCR's detector input is 800x608, so anything much larger than this is downsampled hard

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR)
    return _tokenizer


def count_genie_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip())
    return sentences or [text.strip()]


def _greedy_pack(units: list[str], fits: "callable[[str], bool]", joiner: str = " ") -> list[str]:
    """Greedily pack `units` (sentences) into as few chunks as possible such
    that `fits(chunk)` holds for every chunk. A single unit that alone fails
    `fits` becomes its own (oversized) chunk rather than being silently
    dropped or split mid-sentence."""
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = joiner.join([*current, unit]) if current else unit
        if current and not fits(candidate):
            chunks.append(joiner.join(current))
            current = [unit]
        else:
            current = [unit] if not current else [*current, unit]
    if current:
        chunks.append(joiner.join(current))
    return chunks


def chunk_text_for_genie(text: str, max_tokens: int = DEFAULT_MAX_GENIE_TOKENS) -> list[str]:
    """Split text into windows whose Genie token count each stays under
    max_tokens, packing greedily on sentence/paragraph boundaries. Joins
    packed units with the same "\\n\\n" that _split_sentences splits
    paragraphs on (rather than _greedy_pack's default single space) so that
    OCR row/paragraph boundaries survive packing -- a doctor's-note table row
    rarely ends in sentence punctuation, so each row becomes its own
    paragraph-unit here, and callers that need to recover individual rows
    from a packed chunk (e.g. splitting on blank lines) depend on those
    boundaries not being flattened away."""
    if count_genie_tokens(text) <= max_tokens:
        return [text]
    sentences = _split_sentences(text)
    return _greedy_pack(sentences, fits=lambda c: count_genie_tokens(c) <= max_tokens, joiner="\n\n")


def chunk_text_for_tts(text: str, max_phones: int = DEFAULT_MAX_TTS_PHONES) -> list[str]:
    """Split text into windows whose exact MeloTTS phone count (count_melo_phones)
    each stays under max_phones, packing greedily on sentence boundaries.
    count_melo_phones's BERT feature extraction has its own hard ceiling
    (512 position embeddings) well above max_phones -- text long enough to
    hit that raises a RuntimeError from inside transformers rather than
    returning a phone count, so _fits treats that as "doesn't fit" (forcing
    a further split) instead of letting the exception propagate."""

    def _fits(candidate: str) -> bool:
        try:
            return count_melo_phones(candidate) <= max_phones
        except RuntimeError:
            return False

    if _fits(text):
        return [text]
    sentences = _split_sentences(text)
    return _greedy_pack(sentences, fits=_fits)


def tile_image_if_large(
    image: Image.Image,
    max_dim: int = DEFAULT_MAX_IMAGE_DIM,
    overlap: int = 100,
) -> list[tuple[Image.Image, tuple[int, int]]]:
    """Return [(image, (0, 0))] unchanged if both dimensions are within
    max_dim. Otherwise split into overlapping tiles no larger than max_dim on
    a side, returning each tile alongside its (x_offset, y_offset) in the
    original image (offsets are for future box-level remapping; v1 callers
    just concatenate each tile's OCR text)."""
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return [(image, (0, 0))]

    stride = max_dim - overlap
    x_starts = list(range(0, max(width - max_dim, 0) + 1, stride)) or [0]
    y_starts = list(range(0, max(height - max_dim, 0) + 1, stride)) or [0]
    if x_starts[-1] + max_dim < width:
        x_starts.append(width - max_dim)
    if y_starts[-1] + max_dim < height:
        y_starts.append(height - max_dim)

    tiles = []
    for y in y_starts:
        for x in x_starts:
            box = (x, y, min(x + max_dim, width), min(y + max_dim, height))
            tiles.append((image.crop(box), (x, y)))
    return tiles


def dedupe_ocr_lines(
    blocks: list[list[tuple[str, float, tuple[int, int, int, int] | None]]],
) -> list[tuple[str, float, tuple[int, int, int, int] | None]]:
    """Flatten OCR (text, confidence, box) lines from multiple tiles/pages,
    dropping consecutive duplicate lines that arise from overlapping tile
    regions. box is (xmin, xmax, ymin, ymax) in original-image pixel
    coordinates (the caller is responsible for shifting each tile's boxes by
    that tile's offset before calling this)."""
    seen_last = None
    lines: list[tuple[str, float, tuple[int, int, int, int] | None]] = []
    for block in blocks:
        for text, confidence, box in block:
            stripped = text.strip()
            if not stripped or stripped == seen_last:
                continue
            lines.append((stripped, confidence, box))
            seen_last = stripped
    return lines


def group_lines_into_rows(
    lines: list[tuple[str, float | None, tuple[int, int, int, int] | None]],
    ycenter_ths: float = 0.5,
) -> list[list[tuple[str, float | None, tuple[int, int, int, int] | None]]]:
    """Group OCR lines into rows by y-center overlap (the same
    running-mean-of-the-row-so-far heuristic easyocr.utils.group_text_box
    uses to merge multi-column text into one line), then order each row
    left-to-right by x_min, so a table's columns survive as one coherent row
    instead of scrambled disconnected lines. box is (xmin, xmax, ymin, ymax);
    lines with box=None (human-corrected text has none) aren't grouped --
    each becomes its own row, in input order."""
    if any(box is None for _text, _confidence, box in lines):
        return [[line] for line in lines]

    sorted_lines = sorted(lines, key=lambda line: 0.5 * (line[2][2] + line[2][3]))
    rows: list[list[tuple[str, float | None, tuple[int, int, int, int] | None]]] = []
    current: list[tuple[str, float | None, tuple[int, int, int, int] | None]] = []
    for line in sorted_lines:
        _text, _confidence, box = line
        y_center = 0.5 * (box[2] + box[3])
        if current:
            mean_ycenter = sum(0.5 * (b[2] + b[3]) for _t, _c, b in current) / len(current)
            mean_height = sum(b[3] - b[2] for _t, _c, b in current) / len(current)
            if abs(mean_ycenter - y_center) < ycenter_ths * mean_height:
                current.append(line)
                continue
            rows.append(sorted(current, key=lambda l: l[2][0]))
        current = [line]
    if current:
        rows.append(sorted(current, key=lambda l: l[2][0]))
    return rows
