"""Lightweight, dependency-free entry point to the LMA descriptor.

Any consumer — the WHAM suggestive-motion pipeline, the NLF/MoGe dance pipeline,
or a test — can import the data-producing extractor here WITHOUT pulling in the
pose/depth/training stack (torch, cv2, MoGe, matplotlib). The feature math lives
in ``utils/lma_extractor.py``; this module is the thin interface around it that
``process_lma_features.py`` re-exports and external drivers call.
"""
import numpy as np

from utils.lma_extractor import LMAExtractor


class IdentityFloor:
    """Mock floor for consumers whose joints are already ground-aligned (e.g. WHAM)."""

    def predict(self, z):
        z = np.asarray(z)
        return np.zeros(z.shape[0])


def compute_lma_descriptor(joints, volumes, floors, fps,
                           window_size=55, short_window=5, apply_smoothing=False):
    """The frozen, data-producing path: posed joints + per-frame body volumes +
    per-frame floor models -> the per-frame LMA descriptor that generated the
    published features. Returns ``(feature_dict, (T, 61) matrix)`` with columns in
    sorted-key order.
    """
    extractor = LMAExtractor(window_size=window_size, fps=fps,
                             short_window=short_window, apply_smoothing=apply_smoothing)
    lma_dict = extractor.extract_all_features(joints, volumes, floors)
    keys = sorted(lma_dict)
    matrix = np.stack([lma_dict[k] for k in keys], axis=1)
    return lma_dict, matrix
