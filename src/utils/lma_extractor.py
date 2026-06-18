import numpy as np
from scipy.spatial import ConvexHull


class LMAExtractor:
    def __init__(self, window_size=55, fps=60, apply_smoothing=False, derivative="gradient"):
        """
        Laban Movement Analysis Feature Extractor.
        Faithfully implements the 55-feature vector described in Turab et al. (2025),
        incorporating specific lag-based Space metrics and threshold-based Initiation.

        The per-joint dynamics (velocity / KE / acceleration / jerk) use either
        derivative='gradient' (np.gradient chain, lag-1) or derivative='central'.

        Faithfulness note: Turab et al. (arXiv:2504.21154 §3.3 / 2504.21166 §3.3) name v / a
        only as "the velocity / acceleration of a joint" and give NO finite-difference
        formula; their window w is the Initiation (Eq.1) / Space (Eq.2) lag and the sliding-
        aggregation window, NOT a derivative stencil. The faithful reading of an unspecified
        per-frame derivative is the lag-1 successive difference -> derivative='gradient' (the
        default, and what reproduces the papers). derivative='central' is a NON-PAPER lag-w
        scheme (from the compute_55-10.pdf reference doc, not the papers), kept for comparison
        only — it corresponds to no equation in either paper.
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.apply_smoothing = apply_smoothing
        assert derivative in ("gradient", "central"), \
            f"derivative must be 'gradient' or 'central', got {derivative!r}"
        self.derivative = derivative

        # JA: From MPII:
        # https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/_base_/datasets/mpii.py
        # joint_weights=[
        #     1.5, # right_ankle
        #     1.2, # right_knee
        #     1.,  # right_hip
        #     1.,  # left_hip
        #     1.2, # left_knee
        #     1.5, # left_ankle
        #     1.,  # pelvis
        #     1.,  # thorax
        #     1.,  # upper_neck
        #     1.,  # head_top
        #     1.5, # right_wrist
        #     1.2, # right_elbow
        #     1.,  # right_shoulder
        #     1.,  # left_shoulder
        #     1.2, # left_elbow
        #     1.5  # left_wrist
        # ],

        # Standard SMPL 24-joint topology
        self.IDX = {
            "PELVIS": 0,
            "L_HIP": 1,
            "R_HIP": 2,
            "SPINE1": 3,
            "L_KNEE": 4, # 1.2
            "R_KNEE": 5, # 1.2
            "SPINE2": 6,
            "L_ANKLE": 7, # 1.5
            "R_ANKLE": 8, # 1.5
            "SPINE3": 9,
            "L_FOOT": 10, # 1.5 (from L_ANKLE)
            "R_FOOT": 11, # 1.5 (from R_ANKLE)
            "NECK": 12,
            "L_COLLAR": 13,
            "R_COLLAR": 14,
            "HEAD": 15,
            "L_SHOULDER": 16,
            "R_SHOULDER": 17,
            "L_ELBOW": 18, # 1.2
            "R_ELBOW": 19, # 1.2
            "L_WRIST": 20, # 1.5
            "R_WRIST": 21, # 1.5
            "L_HAND": 22, # 1.5 (from L_WRIST)
            "R_HAND": 23, # 1.5 (from R_WRIST)
        }

        # Turab's 6 key joints, per the SHAP figures (head, pelvis, wrists, ankles).
        # The per-joint dynamics/Effort and the Initiation events share this set.
        self.BODY_KEY_JOINTS = [
            "HEAD", "PELVIS",
            "L_WRIST", "R_WRIST",
            "L_ANKLE", "R_ANKLE"
        ]
        self.EFFORT_KEY_JOINTS = self.BODY_KEY_JOINTS

        # Per-joint weights translated from MPII (mmpose) to SMPL's 24-joint topology.
        # MPII extremities (wrists, ankles) = 1.5; mid-limb (elbows, knees) = 1.2;
        # everything else = 1.0. SMPL adds joints with no direct MPII equivalent —
        # hands inherit from wrists, feet inherit from ankles, collars/spines default to 1.0.
        self.weights = {
            "PELVIS":     1.0,
            "L_HIP":      1.0, "R_HIP":      1.0,
            "SPINE1":     1.0, "SPINE2":     1.0, "SPINE3":     1.0,
            "L_KNEE":     1.2, "R_KNEE":     1.2,
            "L_ANKLE":    1.5, "R_ANKLE":    1.5,
            "L_FOOT":     1.5, "R_FOOT":     1.5,   # inherit from ankle
            "NECK":       1.0,
            "L_COLLAR":   1.0, "R_COLLAR":   1.0,
            "HEAD":       1.0,
            "L_SHOULDER": 1.0, "R_SHOULDER": 1.0,
            "L_ELBOW":    1.2, "R_ELBOW":    1.2,
            "L_WRIST":    1.5, "R_WRIST":    1.5,
            "L_HAND":     1.5, "R_HAND":     1.5,   # inherit from wrist
        }

    def _impute_missing_data(self, all_frames):
        """Linearly interpolates missing frames to ensure continuity."""
        n_frames = len(all_frames)
        valid_indices = [i for i, x in enumerate(all_frames) if len(x) > 0]

        if not valid_indices:
            return np.zeros((n_frames, 24, 3))

        full_seq = np.zeros((n_frames, 24, 3))

        # Fill known values
        for i in valid_indices:
            full_seq[i] = all_frames[i]

        # Interpolate gaps
        for j in range(24):
            for c in range(3):
                vals = full_seq[valid_indices, j, c]
                full_seq[:, j, c] = np.interp(range(n_frames), valid_indices, vals)
        return full_seq

    def _normalize_pose_to_floor(self, joints, floor_models):
        """
        Converts Camera Space -> Floor-Relative Height.
        [cite_start]Crucial for 'Floor Aware Body Modeling'[cite: 83].
        """
        normalized = np.copy(joints)
        n_frames = len(joints)

        for i in range(n_frames):
            z_vals = joints[i, :, 2].reshape(-1, 1)
            try:
                floor_y = floor_models[i].predict(z_vals)
            except Exception:
                # Fallback: assume floor is 1 meter below root if model fails
                floor_y = joints[i, :, 1] + 1.0

            # Y-down coordinate system assumption (common in OpenCV/SMPL)
            normalized[i, :, 1] = floor_y - joints[i, :, 1]

        return normalized

    def extract_all_features(self, all_frames, all_floor_models):
        """
        Aggregate (1,55) descriptor: the time-mean of the per-frame stream, which is the
        single source of truth (extract_per_frame_features). Clip-level features (Directness,
        Effort_Space_Global, Initiation, Traj_*) are constant per frame, so their mean
        returns them unchanged — the per-frame column mean equals the aggregate scalar.
        """
        per_frame = self.extract_per_frame_features(all_frames, all_floor_models)
        return {k: float(np.mean(v)) for k, v in per_frame.items()}

    def extract_per_frame_features(self, all_frames, all_floor_models):
        """
        Per-FRAME (T,55) LMA stream — the SINGLE SOURCE OF TRUTH for the descriptor; the
        aggregate (1,55) is just its time-mean (see extract_all_features). Time-mean
        features (distances, dispersions, body_volume, per-joint vel/KE/Accel/Jerk and the
        Weight/Time/Flow globals) vary per frame; clip-level features (Directness,
        Effort_Space_Global, Initiation, Traj_*) are computed once and broadcast as a
        constant so their column mean is exact. Dynamic lag-w terms are edge-padded at the
        boundaries. Derivative convention per self.derivative (the paper's Eqs. 6 & 9 for
        'central', or np.gradient for 'gradient').
        """
        norm_frames = self._normalize_pose_to_floor(
            self._impute_missing_data(all_frames), all_floor_models)
        n_frames = len(all_frames)
        window_size = self.window_size
        # Floor: the inherently forward Initiation and backward Effort-Space sums (lag w)
        # need at least one valid frame, i.e. n_frames > window_size.
        assert n_frames > window_size, (
            f"Video too short: needs n_frames > {window_size} (window_size), got {n_frames}."
        )
        kj = self.EFFORT_KEY_JOINTS
        effort_indices = [self.IDX[j] for j in kj]
        pf = {}

        def align(vals, start):
            """Place a shorter valid array at [start, start+len) and edge-pad to length T."""
            out = np.empty(n_frames, dtype=float)
            vals = np.asarray(vals, dtype=float)
            m = len(vals)
            if m == 0:
                return np.zeros(n_frames)
            out[start:start + m] = vals
            if start > 0:
                out[:start] = vals[0]
            if start + m < n_frames:
                out[start + m:] = vals[-1]
            return out

        def dist_pf(k1, k2):
            """Per-frame Euclidean distance between two joints."""
            a = norm_frames[:, self.IDX[k1], :]
            b = norm_frames[:, self.IDX[k2], :]
            return np.linalg.norm(a - b, axis=1)

        def const(x):
            return np.full(n_frames, float(x), dtype=float)

        # BODY distances (6) — Euclidean distance between landmark pairs (doc Eq. 11).
        pf["Dist_Hand_Shoulder_L"] = dist_pf("L_WRIST", "L_SHOULDER")
        pf["Dist_Hand_Shoulder_R"] = dist_pf("R_WRIST", "R_SHOULDER")
        pf["Dist_Ankle_Knee_L"]    = dist_pf("L_ANKLE", "L_KNEE")
        pf["Dist_Ankle_Knee_R"]    = dist_pf("R_ANKLE", "R_KNEE")
        pf["Dist_Hands"]           = dist_pf("L_WRIST", "R_WRIST")
        pf["Dist_Feet"]            = dist_pf("L_ANKLE", "R_ANKLE")

        # SPACE dispersions (5) — head/wrists vs torso (SPINE2), ankles vs pelvis (doc §4.1).
        pf["Dispersion_Head"]    = dist_pf("HEAD", "SPINE2")
        pf["Dispersion_R_Wrist"] = dist_pf("R_WRIST", "SPINE2")
        pf["Dispersion_L_Wrist"] = dist_pf("L_WRIST", "SPINE2")
        pf["Dispersion_R_Ankle"] = dist_pf("R_ANKLE", "PELVIS")
        pf["Dispersion_L_Ankle"] = dist_pf("L_ANKLE", "PELVIS")

        # BODY angles (6) — inter-joint angles at limb/torso vertices (Turab §3.3: "Euclidean
        # distance AND angles between the hands, shoulders, pelvis, knees, and ankles").
        # Per-frame; window mean == aggregate. [61-D variant C: A's 55 + these 6 angles.]
        def angle_pf(ka, kb, kc):
            a = norm_frames[:, self.IDX[ka]]; b = norm_frames[:, self.IDX[kb]]; c = norm_frames[:, self.IDX[kc]]
            u = a - b; v = c - b
            nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
            out = np.zeros(n_frames); valid = (nu > 1e-9) & (nv > 1e-9)
            cos = np.sum(u[valid] * v[valid], axis=1) / (nu[valid] * nv[valid])
            out[valid] = np.arccos(np.clip(cos, -1.0, 1.0))
            return out
        pf["Angle_LArm"]      = angle_pf("L_WRIST", "L_SHOULDER", "PELVIS")
        pf["Angle_RArm"]      = angle_pf("R_WRIST", "R_SHOULDER", "PELVIS")
        pf["Angle_Shoulders"] = angle_pf("L_SHOULDER", "PELVIS", "R_SHOULDER")
        pf["Angle_LKnee"]     = angle_pf("PELVIS", "L_KNEE", "L_ANKLE")
        pf["Angle_RKnee"]     = angle_pf("PELVIS", "R_KNEE", "R_ANKLE")
        pf["Angle_Hips"]      = angle_pf("L_KNEE", "PELVIS", "R_KNEE")

        # SHAPE (1) — ConvexHull volume of the 24 joints per frame (doc §5).
        vol = np.zeros(n_frames)
        for i in range(n_frames):
            try:
                vol[i] = ConvexHull(norm_frames[i]).volume
            except Exception:
                # Degenerate frame (collinear/coplanar joints, NaNs): leave at 0.0
                vol[i] = 0.0
        pf["body_volume"] = vol

        # EFFORT velocity / acceleration / jerk for the 6 key joints, per self.derivative.
        # 'gradient' (PAPER-FAITHFUL, default): lag-1 successive differences via np.gradient —
        #   the minimal reading of the papers' unspecified "velocity / acceleration of a joint"
        #   (Turab Eqs. 4-5), aggregated over the window w. This is what reproduces the papers.
        # 'central' (NON-PAPER): the compute_55-10.pdf reference doc's lag-w finite differences
        #   — a derivation the papers do not contain (they never use w as a derivative stencil;
        #   see class docstring). Reuses w = 2*w_half frames as the stencil width:
        #     velocity  ref-doc Eq. 6 (central lag w/2; forward/backward over w at the edges),
        #     accel     ref-doc Eq. 9 (central lag w; one-sided from the velocity field),
        #     jerk      3rd central difference [P(t±w/2), P(t±3w/2)] / (w·τf)^3 (one-sided
        #               from the acceleration field at the edges).
        #   Every central denominator spans (2*w_half)*self.dt; full length, no data loss;
        #   source indices are clamped so short clips degrade instead of crashing. Kept for
        #   comparison only — empirically it underperforms 'gradient' on the 4-way task.
        w_half = window_size // 2
        ef = norm_frames[:, effort_indices]
        n = ef.shape[0]
        if self.derivative == "central":
            d1 = (2 * w_half) * self.dt         # w·τf      — velocity / accel / jerk span
            d2 = d1 ** 2                        # (w·τf)^2  — central acceleration
            d3 = d1 ** 3                        # (w·τf)^3  — central jerk
            da = w_half * self.dt               # (w/2)·τf  — one-sided boundary steps

            ar = np.arange(n)
            def shift(arr, k):
                """arr shifted by k frames along time, indices clamped to [0, n-1]."""
                return arr[np.clip(ar + k, 0, n - 1)]

            # velocity: central lag-w/2 interior, forward/backward over w at the edges (Eq. 6)
            velocity = (shift(ef, w_half) - shift(ef, -w_half)) / d1
            v_fwd = (shift(ef, 2 * w_half) - ef) / d1
            v_bwd = (ef - shift(ef, -2 * w_half)) / d1
            m_fwd, m_bwd = ar < w_half, ar > n - 1 - w_half
            velocity[m_fwd], velocity[m_bwd] = v_fwd[m_fwd], v_bwd[m_bwd]

            # acceleration: central lag-w interior, one-sided from velocity at the edges (Eq. 9)
            acceleration = (shift(ef, 2 * w_half) - 2 * ef + shift(ef, -2 * w_half)) / d2
            a_fwd = (shift(velocity, w_half) - velocity) / da
            a_bwd = (velocity - shift(velocity, -w_half)) / da
            m_fwd, m_bwd = ar < 2 * w_half, ar > n - 1 - 2 * w_half
            acceleration[m_fwd], acceleration[m_bwd] = a_fwd[m_fwd], a_bwd[m_bwd]

            # jerk: 3rd central difference interior, one-sided from acceleration at the edges
            jerk = (shift(ef, 3 * w_half) - 3 * shift(ef, w_half)
                    + 3 * shift(ef, -w_half) - shift(ef, -3 * w_half)) / d3
            j_fwd = (shift(acceleration, w_half) - acceleration) / da
            j_bwd = (acceleration - shift(acceleration, -w_half)) / da
            m_fwd, m_bwd = ar < 3 * w_half, ar > n - 1 - 3 * w_half
            jerk[m_fwd], jerk[m_bwd] = j_fwd[m_fwd], j_bwd[m_bwd]

            start = 0
        else:  # 'gradient'
            velocity = np.gradient(norm_frames[:, effort_indices], self.dt, axis=0)
            acceleration = np.gradient(velocity, self.dt, axis=0)
            jerk = np.gradient(acceleration, self.dt, axis=0)
            start = 0

        # Per-joint Kinematics velocity + Effort Weight=KE / Time=‖a‖ / Flow=‖jerk‖, and the
        # per-frame Weight/Time/Flow globals as Σ_j α_j·(per-joint value) (doc Eqs. 18-24).
        v_mag = np.linalg.norm(velocity, axis=2)            # (L, 6)
        a_mag = np.linalg.norm(acceleration, axis=2)
        j_mag = np.linalg.norm(jerk, axis=2)
        ke = 0.5 * np.sum(velocity ** 2, axis=2)            # ½‖v‖²
        g_weight = np.zeros(n_frames)
        g_time = np.zeros(n_frames)
        g_flow = np.zeros(n_frames)
        for i, joint_name in enumerate(kj):
            a = self.weights[joint_name]
            pf[f"{joint_name}_vel"]   = align(v_mag[:, i], start)
            pf[f"{joint_name}_KE"]    = align(ke[:, i], start)
            pf[f"{joint_name}_Accel"] = align(a_mag[:, i], start)
            pf[f"{joint_name}_Jerk"]  = align(j_mag[:, i], start)
            g_weight += a * pf[f"{joint_name}_KE"]
            g_time   += a * pf[f"{joint_name}_Accel"]
            g_flow   += a * pf[f"{joint_name}_Jerk"]
        pf["Effort_Weight_Global"] = g_weight
        pf["Effort_Time_Global"]   = g_time
        pf["Effort_Flow_Global"]   = g_flow

        # EFFORT Space (directness) per joint: Σ ||P(t)-P(t-w)|| / ||P(T)-P(0)|| (doc Eq. 17,
        # lag w) — clip-level ratios broadcast as constants; their weighted sum is the global.
        g_space = 0.0
        for joint_name in kj:
            idx = self.IDX[joint_name]
            diffs = norm_frames[window_size:, idx] - norm_frames[:n_frames - window_size, idx]
            total_displacement = float(np.sum(np.linalg.norm(diffs, axis=1)))
            net_displacement = np.linalg.norm(norm_frames[n_frames - 1, idx] - norm_frames[0, idx]) + 1e-9
            directness = total_displacement / net_displacement
            pf[f"{joint_name}_Directness"] = const(directness)
            g_space += self.weights[joint_name] * directness
        pf["Effort_Space_Global"] = const(g_space)

        # BODY Initiation (6): proportion of valid frames whose forward lag-w speed exceeds
        # sigma (doc Eq. 16) — a clip-level rate, broadcast as a constant.
        for joint_name in kj:
            idx = self.IDX[joint_name]
            speeds = np.linalg.norm(
                norm_frames[window_size:, idx] - norm_frames[:n_frames - window_size, idx],
                axis=1) / (window_size * self.dt)
            pf[f"Initiation_{joint_name}"] = const(float(np.mean(speeds > np.std(speeds))))

        # SPACE Trajectory (3) — pelvis path length / net displacement / their ratio
        # (doc §4.2, Eqs. 27-28) — clip-level, broadcast.
        pelvis_traj = norm_frames[:, self.IDX["PELVIS"], :]
        path = float(np.sum(np.linalg.norm(pelvis_traj[1:] - pelvis_traj[:-1], axis=1)))
        disp = float(np.linalg.norm(pelvis_traj[-1] - pelvis_traj[0]))
        pf["Traj_Path_Length"]  = const(path)
        pf["Traj_Displacement"] = const(disp)
        pf["Traj_Curvature"]    = const(path / (disp + 1e-6))

        return pf
