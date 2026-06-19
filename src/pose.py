"""Pluggable 3D pose estimation.

The pipeline needs, per frame, the detected person's 3D joints and mesh vertices. WHICH
model produces them is a swappable choice:

  * ``NLFPoseEstimator`` — Neural Localizer Fields (Sarandi), the original *Dance Style
    Recognition* pipeline's per-frame pose backend. The torchscript checkpoint is the heavy
    dependency and is loaded lazily, so this module imports without it.

To customize, pass your own estimator (anything with ``estimate(frame) -> (joints3d,
vertices3d)``) to ``process_lma_features.process_single_video(..., pose_estimator=...)``, or
implement ``PoseEstimator`` for a new backend. The suggestive-motion paper does NOT use this
path at all — it gets pose from WHAM per video (see the parent repo's core/wham_inference.py)
and feeds the resulting joints straight to the LMA descriptor.
"""


class PoseEstimator:
    """Base class. ``estimate(frame)`` returns ``(joints3d, vertices3d)`` for the detected
    person in a single BGR frame."""

    def estimate(self, frame):
        raise NotImplementedError


class NLFPoseEstimator(PoseEstimator):
    """Per-frame 3D pose from Neural Localizer Fields (the dance pipeline's default).

    Pass an already-loaded torchscript ``model`` to reuse it, or leave it ``None`` and the
    checkpoint at ``model_path`` is loaded lazily on first use (so importing this module
    never loads the model).
    """

    def __init__(self, model=None, device="cuda",
                 model_path="models/nlf_l_multi_0.3.2.torchscript"):
        self._model = model
        self.device = device
        self.model_path = model_path

    def _ensure_model(self):
        if self._model is None:
            import torch
            self._model = torch.jit.load(self.model_path).to(self.device).eval()
        return self._model

    def estimate(self, frame):
        import torch

        model = self._ensure_model()
        # BGR -> RGB, HWC -> CHW, [0,1]
        frame_tensor = torch.from_numpy(frame[..., ::-1].copy()).to(self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1).float() / 255.0
        frame_batch = frame_tensor.unsqueeze(0)
        with torch.inference_mode():
            pred = model.detect_smpl_batched(frame_batch)
        return pred["joints3d"], pred["vertices3d"]
