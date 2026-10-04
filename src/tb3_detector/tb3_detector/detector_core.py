#!/usr/bin/env python3
"""
detector_core.py  —  Stage-1 perception: YOLO26 inference wrapper.

████████████████████████████████████████████████████████████████████████████
██                                                                        ██
██   STUDENT ASSIGNMENT — THIS FILE IS THE CORE OF YOUR TASK              ██
██                                                                        ██
██   The rest of the navigation stack (localizer, memory, query, Nav2)    ██
██   is provided and working. It is waiting for real detections from      ██
██   this class. Until you implement load() and infer(), the robot        ██
██   explores and maps fine, but every "go to person 0" command answers   ██
██   "query failed: no active person in memory".                          ██
██                                                                        ██
██   Read INSTRUCTIONS.md at the repository root before you start.        ██
██                                                                        ██
████████████████████████████████████████████████████████████████████████████

Responsibilities of this class:
  - Load a YOLO26 model from a configurable local path.
  - Run inference on a BGR numpy image (from cv_bridge).
  - Return a list of Detection dicts (format below).

NOT responsible for:
  - Coordinate projection to 3D (→ tb3_localizer, provided)
  - Maintaining object history  (→ tb3_memory,    provided)
  - Answering semantic queries  (→ tb3_query,     provided)
  - Sending Nav2 goals          (→ tb3_nav_adapter / tb3_coordinator, provided)

============================================================================
OUTPUT CONTRACT — do not change key names; the provided localizer and the
node wrapper (detector_node.py) rely on them:

    {
        "label":      str,          # detector class name: "person", "trash_can", "chair"
        "conf":       float,        # confidence, 0.0 – 1.0
        "bbox_xyxy":  [x1, y1, x2, y2],   # pixels (float), see convention below
        "track_id":   int | None,   # None unless you enable tracking
    }

Pixel-coordinate convention (standard image coordinates):
    - origin (0, 0) is the TOP-LEFT corner of the image
    - x grows to the RIGHT, y grows DOWN
    - (x1, y1) = top-left corner of the box, (x2, y2) = bottom-right corner
    - values are in pixels of the ORIGINAL /camera/image_raw resolution —
      if you resize the image for inference, scale the boxes back!
      (ultralytics already returns boxes in original-image pixels.)
    The provided localizer uses the bbox centre x = (x1+x2)/2 to compute a
    bearing through the camera FOV, so a wrong x coordinate sends the robot
    in a wrong direction.
============================================================================
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — graceful failure if ultralytics is not installed.
# Keeping this guarded means the package still *builds* without the Python
# dependencies; load() is where the missing dependency becomes a hard error.
#
#     pip install 'ultralytics==8.4.31'   # pinned; yolo26n needs >= 8.4.x
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO as _UltralyticsYOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _UltralyticsYOLO = None
    _ULTRALYTICS_AVAILABLE = False
    logger.warning(
        "ultralytics not found. Install with:  pip install 'ultralytics==8.4.31'\n"
        "detector_core will raise RuntimeError on load() until then."
    )


# Keys every downstream consumer may rely on:
DETECTION_KEYS = ("label", "conf", "bbox_xyxy", "track_id")


class DetectorCore:
    """
    Thin wrapper around a YOLO26 model.

    Usage::

        core = DetectorCore(model_path="models/tb3det_yolo26n.pt", conf_threshold=0.40)
        core.load()                          # loads weights once at startup
        detections = core.infer(bgr_image)   # list of dicts (see module docstring)

    Parameters
    ----------
    model_path : str | Path
        Absolute or relative path to the YOLO26 .pt weights file.
    conf_threshold : float
        Minimum confidence to include a detection (0.0 – 1.0).
        The shipped config uses 0.40 for the fine-tuned tb3det_yolo26n weights
        (threshold sweep in config/detector.yaml). Tune it there, not here.
    class_filter : list[str] | None
        If given, only return detections whose label is in this list.
        None means return all detected classes.
        IMPORTANT: entries are *detector labels* as the weights emit them
        ("person", "trash_can", "chair" for the shipped weights; the COCO
        weights say "traffic light" for the trash can), NOT the task-level
        semantic names — see the mapping note below.
    device : str
        Torch device string, e.g. "cpu", "cuda:0".
    enable_tracking : bool
        If True, use model.track() instead of model.predict() (ByteTrack).
        Optional — the course task only requires plain per-frame detection.

    ------------------------------------------------------------------------
    Detector label → task label mapping (important!)
    ------------------------------------------------------------------------
    The mapping used by the rest of the stack lives in
    src/tb3_bringup/config/semantic_targets.yaml:

        task name    detector_label     Gazebo model
        ---------    --------------     ------------
        person   ←   "person"           person_standing
        trash_can ←  "trash_can"        first_2015_trash_can
        chair    ←   "chair"            VisitorChair

    With the shipped fine-tuned weights the two columns agree. They did not
    with the COCO weights (no trash-can class; yolo26n called that model a
    *traffic light*), and that is why detector_label and semantic_name are
    separate fields: DetectorCore reports whatever the network says, and the
    provided downstream nodes translate it using semantic_targets.yaml. Do not
    rename labels inside infer().
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        class_filter: list[str] | None = None,
        device: str = "cpu",
        enable_tracking: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = float(conf_threshold)
        self.class_filter = set(class_filter) if class_filter else None
        self.device = device
        self.enable_tracking = enable_tracking

        self._model: Any = None   # set by load()
        self._threads = 0         # optional CPU thread cap, set by load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        """
        Load model weights.  Called once by detector_node at startup.

        ████ TODO(student) ── Step 1: make sure the weights are in place, Step 2: load them ████

        1. The weights are already here — nothing to download. The fine-tuned
           YOLO26 model (models/tb3det_yolo26n.pt, 5.4 MB, labels "person",
           "trash_can", "chair") ships with the repository; it is the one
           *.pt file exempted from the ignore rule in models/.gitignore.
           After a fresh checkout rebuild once so the file is copied into the
           install tree:

               colcon build --symlink-install --packages-select tb3_detector

           or pass an absolute path via model_path. The COCO comparison
           weights are a separate, optional local download (see NOTES.md).

        2. Implement loading here:
             - raise RuntimeError with a helpful message if ultralytics is
               not installed (_ULTRALYTICS_AVAILABLE is False — keep startup
               errors readable for the grader);
             - raise FileNotFoundError if self.model_path is missing;
             - otherwise create the model:  self._model = _UltralyticsYOLO(str(self.model_path))
               and move it to self.device (self._model.to(self.device)).
             - Optional but useful: log the class names once,
               list(self._model.names.values()) — you should see exactly
               ['person', 'trash_can', 'chair'].
        """
        # Fail loudly and readably if a dependency is missing.
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "DetectorCore.load(): the 'ultralytics' package is not installed. "
                "Install it with:  pip install 'ultralytics==8.4.31'"
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"DetectorCore.load(): weights file not found: {self.model_path}. "
                "Rebuild once with  colcon build --symlink-install --packages-select "
                "tb3_detector  so models/tb3det_yolo26n.pt reaches install/, "
                "or pass an absolute model_path."
            )

        # Optional cap on the CPU threads PyTorch uses for inference (applied in
        # infer()). Unset keeps the default behaviour. On a laptop that also runs
        # Gazebo, RViz, SLAM and Nav2, the default can starve the navigation stack.
        raw = os.environ.get("TB3_DETECTOR_THREADS", "").strip()
        try:
            self._threads = int(raw) if raw else 0
        except ValueError:
            raise ValueError(
                f"TB3_DETECTOR_THREADS must be a non-negative integer, got {raw!r}") from None
        if self._threads < 0:
            raise ValueError(f"TB3_DETECTOR_THREADS must be >= 0, got {self._threads}")
        if self._threads:
            logger.warning("DetectorCore.load(): torch CPU threads will be capped at %d "
                           "(TB3_DETECTOR_THREADS)", self._threads)

        # Load the fine-tuned YOLO26 weights and move them to the configured device.
        self._model = _UltralyticsYOLO(str(self.model_path))
        self._model.to(self.device)

        # Warn (do not raise) if the whitelist names a label these weights never
        # emit: such detections would otherwise disappear silently (NOTES.md §1.4).
        names = set(self._model.names.values())
        unknown = (self.class_filter or set()) - names
        if unknown:
            logger.warning(
                "DetectorCore.load(): class_filter %s is not emitted by %s (classes: %s)",
                sorted(unknown), self.model_path.name, sorted(names),
            )

    # ------------------------------------------------------------------
    def infer(self, bgr_image) -> list[dict]:
        """
        Run inference on a single BGR uint8 numpy array (shape H×W×3).

        Returns a (possibly empty) list of detection dicts, format specified
        in the module docstring. Never returns None.

        ████ TODO(student) ── Step 3: implement inference ████

        Outline (ultralytics does most of the work):

            results = self._model.predict(
                bgr_image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
            for result in results:
                for each box in result.boxes:
                    label = result.names[int(box.cls)]     # detector label: "person", "trash_can", "chair"
                    ... skip it if self.class_filter is set and label not in it ...
                    conf  = float(box.conf)
                    xyxy  = box.xyxy[0].tolist()           # [x1, y1, x2, y2] pixels
                    ... append {"label": label, "conf": conf,
                                "bbox_xyxy": xyxy, "track_id": None} ...

        Notes:
          - ultralytics accepts BGR numpy arrays directly; no colour
            conversion or resizing is needed (boxes come back in original
            image pixels).
          - Return [] when nothing is detected — never None.
          - Raise RuntimeError if load() has not been called (self._model is
            None) rather than silently returning [] — a missing model should
            be loud once you have implemented load().
          - Optional: if self.enable_tracking, use self._model.track(...,
            persist=True) and fill "track_id" from box.id.
        """
        if self._model is None:
            raise RuntimeError("DetectorCore.infer() called before load(): no model is loaded.")

        # ultralytics takes the BGR frame as is and returns boxes in original-image pixels.
        if self.enable_tracking:
            results = self._model.track(
                bgr_image, conf=self.conf_threshold, device=self.device,
                persist=True, verbose=False,
            )
        else:
            results = self._model.predict(
                bgr_image, conf=self.conf_threshold, device=self.device, verbose=False,
            )

        # ultralytics resets torch's thread count when it sets up the predictor on the
        # first call, so the optional cap from load() is applied after inference.
        if self._threads:
            import torch
            if torch.get_num_threads() != self._threads:
                torch.set_num_threads(self._threads)

        detections: list[dict] = []
        for result in results:
            for box in result.boxes:
                # Report the label exactly as the network emits it; the mapping to
                # task names happens downstream (semantic_targets.yaml).
                label = result.names[int(box.cls)]
                if self.class_filter is not None and label not in self.class_filter:
                    continue
                track_id = None
                if self.enable_tracking and box.id is not None:
                    track_id = int(box.id)
                detections.append({
                    "label": label,
                    "conf": float(box.conf),
                    "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],  # [x1, y1, x2, y2]
                    "track_id": track_id,
                })
        return detections

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
