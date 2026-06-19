"""Pluggable floor estimation.

The LMA extractor normalizes every pose to a floor-relative height, so it needs a
per-frame "floor model" — any object with ``predict(z) -> floor_y``. HOW that floor is
found is a swappable choice:

  * ``MoGeFloorEstimator`` — fits a floor plane to MoGe metric-depth point clouds
    (the original *Dance Style Recognition* paper's approach). MoGe is the heavy,
    optional dependency; it is imported lazily, so this module — and any consumer that
    only wants a flat floor — loads without MoGe installed.
  * ``FlatFloorEstimator`` — a fixed floor at y=0, for inputs whose joints are already
    ground-aligned (e.g. WHAM output). No depth model required.

To customize, pass your own estimator (anything with ``estimate(frame) -> floor_model``)
to ``process_lma_features.process_single_video(..., floor_estimator=...)``, or implement
``FloorEstimator`` for a new depth/geometry backend.
"""
from lma_descriptor import IdentityFloor

# The flat-floor model (y=0) is the same object the WHAM/suggestive-motion path uses.
FlatFloor = IdentityFloor


class FloorEstimator:
    """Base class. ``estimate(frame)`` returns a per-frame floor model with
    ``predict(z) -> floor_y`` (z = per-joint depth, floor_y = floor height at that depth)."""

    def estimate(self, frame):
        raise NotImplementedError


class FlatFloorEstimator(FloorEstimator):
    """Always returns a flat floor at y=0. Use when joints are already ground-aligned
    (so no depth model is needed) — this is what the WHAM suggestive-motion pipeline uses."""

    def __init__(self):
        self._floor = FlatFloor()

    def estimate(self, frame=None):
        return self._floor


class MoGeFloorEstimator(FloorEstimator):
    """Per-frame floor plane fit from MoGe metric depth (the default for the dance pipeline).

    Pass an already-loaded ``model`` to reuse it across frames, or leave it ``None`` and the
    MoGe checkpoint is fetched lazily on first use (so importing this module never loads MoGe).
    """

    def __init__(self, model=None, device="cuda", quantile=0.95, target_points=2000,
                 model_name="Ruicheng/moge-2-vitl-normal"):
        self._model = model
        self.device = device
        self.quantile = quantile
        self.target_points = target_points
        self.model_name = model_name

    def _ensure_model(self):
        if self._model is None:
            from moge.model.v2 import MoGeModel
            self._model = MoGeModel.from_pretrained(self.model_name).to(self.device)
        return self._model

    def estimate(self, frame):
        import cv2
        import torch
        from sklearn.linear_model import QuantileRegressor

        model = self._ensure_model()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(img / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        output = model.infer(input_tensor)
        points = output["points"].cpu().numpy()
        valid_points = points[output["mask"].cpu().numpy().astype(bool)]

        if len(valid_points) > self.target_points:
            scene_cloud = valid_points[::len(valid_points) // self.target_points]
        else:
            scene_cloud = valid_points

        # Fit a quantile line to the lowest points (handles a tilted/sloped floor) over the
        # XZ plane: floor height y as a function of depth z. The returned model is the floor.
        qr = QuantileRegressor(quantile=self.quantile, alpha=0, solver="highs")
        qr.fit(scene_cloud[:, 2].reshape(-1, 1), scene_cloud[:, 1])
        return qr
