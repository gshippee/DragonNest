"""Run EasyOCR's detector + recognizer locally via qnn-net-run on the
Snapdragon NPU (or CPU), reusing qai_hub_models' EasyOCRApp for all
pre/post-processing (box detection/grouping, cropping, greedy decoding).

Two interchangeable runtimes for the same underlying network:
- --runtime dlc (default): detector.dlc/recognizer.dlc, float32 I/O, run via
  qnn-net-run.exe (qnn_runner.py).
- --runtime onnx: detector.onnx/recognizer.onnx, uint8 w8a8-quantized QDQ
  I/O, run via ONNX Runtime's QNN EP (onnxrt_runner.py). The QDQ graph's
  Sub node (confirmed via onnx.load()) subtracts the ImageNet mean
  [0.485, 0.456, 0.406] internally and folds the std-division into its
  quantization scale, so OnnxModule feeds plain [0,1] RGB/grayscale --
  same convention as DlcModule -- with no extra normalization.

EasyOCRApp.recognizer_inference() (qai_hub_models' vendored code) calls the
recognizer once per detected text box in a Python loop -- each call is a
separate qnn-net-run.exe process spawn + graph load + DSP-session-open, which
dominates wall-clock time for any note with more than a couple of lines. This
module monkeypatches build_app()'s EasyOCRApp instance with
_recognizer_inference_batched() (below), which is a byte-for-byte copy of
qai_hub_models' recognizer_inference() except the per-cutout self.recognizer(x)
loop is replaced with one DlcModule.call_batch()/OnnxModule.call_batch() call
that runs every cutout through a single qnn-net-run.exe invocation.

On top of that, --runtime dlc --backend htp loads precompiled HTP context
binaries (context_cache/*.bin, produced by prepare_context_binaries.py)
instead of the raw .dlc files. qnn-net-run.exe recompiles a .dlc's whole graph
(optimization/VTCM-allocation/sequencing passes) on every single invocation --
measured at ~185-200s for recognizer.dlc alone -- which is what actually caused
early recognizer calls to look "hung" against a 120s timeout (the DSP-open
error logged at the same time is a red herring: it falls back to the user
driver path and the run completes normally once the recompile finishes).
Loading a precompiled context binary via --retrieve_context skips all of that
and runs in ~1-2s. ContextBinaryModule below wraps run_context_binary_batch()
for this path; DlcModule (raw --dlc_path) is kept for --backend cpu (context
binaries are backend-specific and a HTP one won't run on QnnCpu.dll) and as
the fallback when no context binary has been generated yet.

EasyOCRApp.predict_text_from_image() also discards each line's detected box
after drawing it onto a debug image, even though recognizer_get_text() (one
call-frame below it) already computes box+text+confidence together. This
module additionally monkeypatches predict_text_from_image with
_predict_text_with_boxes() (below) -- identical except each result also
carries the box, letting doctor_note_pipeline.py reconstruct table rows
(chunking.group_lines_into_rows) instead of treating OCR as a flat list of
disconnected lines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from onnxrt_runner import run_onnx  # noqa: E402
from qnn_runner import run_context_binary_batch, run_dlc_batch  # noqa: E402

from easyocr.recognition import custom_mean  # noqa: E402
from qai_hub_models.models.easyocr.app import RECOGNIZER_ARGS, EasyOCRApp  # noqa: E402
from qai_hub_models.utils.draw import draw_box_from_corners, draw_box_from_xyxy  # noqa: E402
from qai_hub_models.utils.image_processing import app_to_net_image_inputs  # noqa: E402

EASYOCR_DIR = Path(
    r"C:\Users\harisury\Downloads\easyocr-qnn_dlc-float\easyocr-qnn_dlc-float"
)
DETECTOR_DLC = EASYOCR_DIR / "detector.dlc"
RECOGNIZER_DLC = EASYOCR_DIR / "recognizer.dlc"

CONTEXT_BIN_DIR = Path(__file__).parent / "context_cache"
DETECTOR_CTX_BIN = CONTEXT_BIN_DIR / "detector_htp.bin"
RECOGNIZER_CTX_BIN = CONTEXT_BIN_DIR / "recognizer_htp.bin"

EASYOCR_ONNX_DIR = Path(
    r"C:\Users\harisury\Downloads\ONNX\easyocr-onnx-w8a8\easyocr-onnx-w8a8"
)
DETECTOR_ONNX = EASYOCR_ONNX_DIR / "detector.onnx"
RECOGNIZER_ONNX = EASYOCR_ONNX_DIR / "recognizer.onnx"
ONNX_METADATA = json.loads((EASYOCR_ONNX_DIR / "metadata.json").read_text())

DETECTOR_IMG_SHAPE = (608, 800)  # (H, W)
RECOGNIZER_IMG_SHAPE = (64, 800)  # (H, W)


def _quant_params(file_name: str, direction: str, tensor_name: str) -> tuple[float, int]:
    spec = ONNX_METADATA["model_files"][file_name][direction][tensor_name]
    qp = spec["quantization_parameters"]
    return qp["scale"], qp["zero_point"]


class ContextBinaryModule:
    """Callable[[torch.Tensor], torch.Tensor] backed by a precompiled HTP
    context binary (--retrieve_context), the fast counterpart to DlcModule --
    see module docstring for why this exists. Same NCHW-in/NHWC-out contract
    as DlcModule.
    """

    def __init__(
        self,
        bin_path: Path,
        output_name: str,
        output_shape: tuple[int, ...],
        backend: str = "htp",
    ) -> None:
        self.bin_path = bin_path
        self.output_name = output_name
        self.output_shape = output_shape
        self.backend = backend

    def __call__(self, nchw: torch.Tensor) -> torch.Tensor:
        return self.call_batch([nchw])[0]

    def call_batch(self, nchw_list: list[torch.Tensor]) -> list[torch.Tensor]:
        inputs_list = [
            {"image": nchw.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)}
            for nchw in nchw_list
        ]
        outputs_list = run_context_binary_batch(
            self.bin_path,
            inputs_list=inputs_list,
            output_names=[self.output_name],
            output_shapes={self.output_name: self.output_shape},
            output_dtypes={self.output_name: np.float32},
            backend=self.backend,
        )
        return [torch.from_numpy(outputs[self.output_name]) for outputs in outputs_list]


class DlcModule:
    """Callable[[torch.Tensor], torch.Tensor] backed by a compiled .dlc graph.

    Accepts NCHW float32 input (as produced by qai_hub_models' image utils),
    permutes to the NHWC layout the compiled DLC expects, runs it via
    qnn-net-run, and returns the NHWC output as-is (downstream post-processing
    in EasyOCRApp already expects NHWC for both the detector and recognizer
    outputs).
    """

    def __init__(
        self,
        dlc_path: Path,
        output_name: str,
        output_shape: tuple[int, ...],
        backend: str = "htp",
    ) -> None:
        self.dlc_path = dlc_path
        self.output_name = output_name
        self.output_shape = output_shape
        self.backend = backend

    def __call__(self, nchw: torch.Tensor) -> torch.Tensor:
        return self.call_batch([nchw])[0]

    def call_batch(self, nchw_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run many inputs through one qnn-net-run.exe invocation instead of
        one process per input -- see module docstring."""
        inputs_list = [
            {"image": nchw.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)}
            for nchw in nchw_list
        ]
        outputs_list = run_dlc_batch(
            self.dlc_path,
            inputs_list=inputs_list,
            output_names=[self.output_name],
            output_shapes={self.output_name: self.output_shape},
            output_dtypes={self.output_name: np.float32},
            backend=self.backend,
        )
        return [torch.from_numpy(outputs[self.output_name]) for outputs in outputs_list]


class OnnxModule:
    """Callable[[torch.Tensor], torch.Tensor] backed by a QDQ .onnx graph via
    ONNX Runtime + QNN EP. Same NCHW-in/NHWC-out contract as DlcModule, except
    the graph's I/O is uint8 w8a8-quantized -- quantize plain [0,1] float
    input to uint8 using metadata.json's input scale/zero_point before the
    call, dequantize the uint8 output back to float after.
    """

    def __init__(
        self,
        onnx_path: Path,
        file_name: str,
        input_name: str,
        output_name: str,
        backend: str = "htp",
    ) -> None:
        self.onnx_path = onnx_path
        self.input_name = input_name
        self.output_name = output_name
        self.backend = backend
        self.input_scale, self.input_zero_point = _quant_params(
            file_name, "inputs", input_name
        )
        self.output_scale, self.output_zero_point = _quant_params(
            file_name, "outputs", output_name
        )

    def __call__(self, nchw: torch.Tensor) -> torch.Tensor:
        # Unlike the .dlc (NHWC I/O), the ONNX graph's declared input/output
        # shapes (metadata.json) are NCHW -- no permute needed here.
        chw = nchw.contiguous().numpy().astype(np.float32)
        quantized = np.clip(
            np.round(chw / self.input_scale) + self.input_zero_point, 0, 255
        ).astype(np.uint8)
        outputs = run_onnx(
            self.onnx_path,
            inputs={self.input_name: quantized},
            output_names=[self.output_name],
            backend=self.backend,
        )
        dequantized = (
            outputs[self.output_name].astype(np.float32) - self.output_zero_point
        ) * self.output_scale
        return torch.from_numpy(dequantized)

    def call_batch(self, nchw_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # onnxrt_runner's worker process is persistent (model stays loaded),
        # so per-item overhead here is far lower than qnn-net-run's per-call
        # process spawn -- no batching needed to get the same win.
        return [self(nchw) for nchw in nchw_list]


def _recognizer_inference_batched(
    self: EasyOCRApp, cutout_frames: list[torch.Tensor]
) -> list[tuple[str, np.float64]]:
    """Drop-in replacement for EasyOCRApp.recognizer_inference(), bound onto
    the app instance in build_app(). Identical to qai_hub_models' own
    implementation except the per-cutout `self.recognizer(cutout_frame) for
    cutout_frame in cutout_frames` loop is replaced with one
    call_batch(cutout_frames) call."""
    if not cutout_frames:
        return []

    # EasyOCRApp's low-confidence high-contrast retry path (recognizer_get_text)
    # calls TRF.adjust_contrast(img, contrast).unsqueeze(0) on cutouts that are
    # already batched ([1,1,H,W]), producing 5D ([1,1,1,H,W]) tensors -- squeeze
    # back to the 4D shape call_batch expects (identical to the initial pass).
    cutout_frames = [f.squeeze(0) if f.dim() == 5 else f for f in cutout_frames]

    result: list[tuple[str, np.float64]] = []
    with torch.no_grad():
        preds_list = self.recognizer.call_batch(cutout_frames)
    preds = torch.cat(preds_list)

    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = torch.IntTensor([preds.size(1)] * len(cutout_frames))

    ######## filter ignore_char, rebalance
    preds_prob = F.softmax(preds, dim=2)
    preds_prob[:, :, self.ignore_char_idx] = 0.0
    pred_norm = preds_prob.sum(dim=2)
    preds_prob = preds_prob / pred_norm.unsqueeze(-1)

    preds_str: list[str]
    if self.decoder == "greedy":
        _, preds_index = preds_prob.max(2)
        preds_index = preds_index.view(-1)
        preds_str = self.converter.decode_greedy(
            preds_index.data.cpu().detach().numpy(), preds_size.data
        )
    elif self.decoder == "beamsearch":
        k = preds_prob.cpu().detach().numpy()
        preds_str = self.converter.decode_beamsearch(
            k, beamWidth=RECOGNIZER_ARGS["beamWidth"]
        )
    elif self.decoder == "wordbeamsearch":
        k = preds_prob.cpu().detach().numpy()
        preds_str = self.converter.decode_wordbeamsearch(
            k, beamWidth=RECOGNIZER_ARGS["beamWidth"]
        )
    else:
        raise NotImplementedError(f"Unknown decoder {self.decoder}")

    preds_prob_np = preds_prob.cpu().detach().numpy()
    values = preds_prob_np.max(axis=2)
    indices = preds_prob_np.argmax(axis=2)
    preds_max_prob = []
    for v, i in zip(values, indices, strict=False):
        max_probs = v[i != 0]
        if len(max_probs) > 0:
            preds_max_prob.append(max_probs)
        else:
            preds_max_prob.append(np.array([0]))

    for pred, pred_max_prob in zip(preds_str, preds_max_prob, strict=False):
        confidence_score = custom_mean(pred_max_prob)
        result.append((pred, confidence_score))

    return result


def _predict_text_with_boxes(
    self: EasyOCRApp, pixel_values_or_image: np.ndarray | Image.Image
) -> list[tuple[Image.Image, list[str], list[np.float64], list[tuple[int, int, int, int]]]]:
    """Drop-in replacement for EasyOCRApp.predict_text_from_image(), bound
    onto the app instance in build_app(). Byte-for-byte copy of
    qai_hub_models' own implementation except each result also carries the
    box (xmin, xmax, ymin, ymax) that predict_text_from_image discards after
    drawing it onto the debug image -- recognizer_get_text already computes
    and returns this box paired with the text/confidence, one call-frame
    below this. Free (slanted) boxes are converted to their axis-aligned
    (xmin, xmax, ymin, ymax) bounding box so downstream row-grouping (see
    chunking.group_lines_into_rows) has one uniform box shape to work with."""
    NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)
    NHWC_int_numpy_GRAY_frames = [
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in NHWC_int_numpy_frames
    ]

    detector_input_frames, scales, paddings = self.detector_preprocess(NHWC_int_numpy_frames)
    results = self.detector(detector_input_frames)
    horizontal_boxes_per_img, free_boxes_per_img = self.detector_postprocess(
        results, scales, paddings
    )

    output = []
    for img, img_gray, horizontal_boxes, free_boxes in zip(
        NHWC_int_numpy_frames,
        NHWC_int_numpy_GRAY_frames,
        horizontal_boxes_per_img,
        free_boxes_per_img,
        strict=False,
    ):
        img_results_horizontal, img_results_free = self.recognizer_get_text(
            img_gray, horizontal_boxes, free_boxes
        )

        img = img.copy()
        texts, confidences, boxes = [], [], []
        for box_coords, text, confidence in img_results_horizontal:
            draw_box_from_xyxy(
                img,
                (box_coords[0], box_coords[2]),
                (box_coords[1], box_coords[3]),
                color=(0, 255, 0),
                size=2,
            )
            texts.append(text)
            confidences.append(confidence)
            boxes.append((box_coords[0], box_coords[1], box_coords[2], box_coords[3]))

        for free_box_coords, text, confidence in img_results_free:
            draw_box_from_corners(
                img,
                np.array(free_box_coords),
                color=(0, 255, 0),
                size=2,
            )
            texts.append(text)
            confidences.append(confidence)
            xs = [c[0] for c in free_box_coords]
            ys = [c[1] for c in free_box_coords]
            boxes.append((min(xs), max(xs), min(ys), max(ys)))

        output.append((Image.fromarray(img), texts, confidences, boxes))

    return output


def build_app(backend: str = "htp", runtime: str = "dlc") -> EasyOCRApp:
    if runtime == "dlc":
        if backend == "htp" and DETECTOR_CTX_BIN.exists() and RECOGNIZER_CTX_BIN.exists():
            detector = ContextBinaryModule(
                DETECTOR_CTX_BIN, "results", (1, 304, 400, 2), backend=backend
            )
            recognizer = ContextBinaryModule(
                RECOGNIZER_CTX_BIN, "output_preds", (1, 199, 97), backend=backend
            )
        else:
            detector = DlcModule(DETECTOR_DLC, "results", (1, 304, 400, 2), backend=backend)
            recognizer = DlcModule(
                RECOGNIZER_DLC, "output_preds", (1, 199, 97), backend=backend
            )
    elif runtime == "onnx":
        detector = OnnxModule(
            DETECTOR_ONNX, "detector.onnx", "image", "results", backend=backend
        )
        recognizer = OnnxModule(
            RECOGNIZER_ONNX,
            "recognizer.onnx",
            "image",
            "output_preds",
            backend=backend,
        )
    else:
        raise ValueError(f"Unknown runtime: {runtime!r}")

    app = EasyOCRApp(
        detector=detector,
        recognizer=recognizer,
        detector_img_shape=DETECTOR_IMG_SHAPE,
        recognizer_img_shape=RECOGNIZER_IMG_SHAPE,
        lang_list=["en"],
        decoder_mode="greedy",
    )
    app.recognizer_inference = MethodType(_recognizer_inference_batched, app)
    app.predict_text_from_image = MethodType(_predict_text_with_boxes, app)
    return app


def run(
    image_path: str, backend: str = "htp", runtime: str = "dlc"
) -> list[tuple[Image.Image, list[str], list[float], list[tuple[int, int, int, int]]]]:
    app = build_app(backend=backend, runtime=runtime)
    image = Image.open(image_path).convert("RGB")
    return app.predict_text_from_image(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--backend", choices=["cpu", "htp"], default="htp")
    parser.add_argument("--runtime", choices=["dlc", "onnx"], default="dlc")
    parser.add_argument("--output_image", default=None)
    args = parser.parse_args()

    results = run(args.image_path, backend=args.backend, runtime=args.runtime)
    for annotated_image, texts, confidences, boxes in results:
        for text, confidence, box in zip(texts, confidences, boxes):
            print(f"{confidence:.3f}\t{box}\t{text}")
        if args.output_image:
            annotated_image.save(args.output_image)
            print(f"Annotated image saved to {args.output_image}")

