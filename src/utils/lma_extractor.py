import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter

class LMAExtractor:
    def __init__(self, window_size=55, fps=60, apply_smoothing=False):
        """
        Laban Movement Analysis Feature Extractor.
        Faithfully implements the 55-feature vector described in Turab et al. (2025),
        incorporating specific lag-based Space metrics and threshold-based Initiation.
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.apply_smoothing = apply_smoothing

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

        self.BODY_KEY_JOINTS = [
            "PELVIS",
            "L_HAND", "R_HAND",
            "L_SHOULDER", "R_SHOULDER",
            "L_KNEE", "R_KNEE",
            "L_ANKLE", "R_ANKLE"
        ]

        self.EFFORT_KEY_JOINTS = [
            "PELVIS",
            "L_HAND", "R_HAND",
            "L_FOOT", "R_FOOT",
            "HEAD"
        ]

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
        Extracts the 55 LMA features with corrected Equation 1 & 2 logic.
        """
        # 1. Preprocessing
        cleaned_frames = self._impute_missing_data(all_frames)
        norm_frames = self._normalize_pose_to_floor(cleaned_frames, all_floor_models)

        # Commented by JA: This smooth of discrete frames by Savgol filter does the
        # same thing as the sliding window of the LMA paper, so we ignore this step
        # temporarily
        # if self.apply_smoothing:
        #     window_len = int(self.fps / 4)
        #     if window_len % 2 == 0:
        #         window_len += 1
        #     window_len = max(5, window_len)
        #     poly_order = min(3, window_len - 2)

        #     for j in range(24):
        #         for c in range(3):
        #             norm_frames[:, j, c] = savgol_filter(norm_frames[:, j, c], window_len, poly_order, deriv=0)

        # vel = np.gradient(norm_frames, self.dt, axis=0)
        # acc = np.gradient(vel, self.dt, axis=0)
        # jerk = np.gradient(acc, self.dt, axis=0)

        n_frames = len(all_frames)
        window_size = self.window_size

        # JA: From "Dance Style Recognition Using Laban Movement Analysis":
        # To quantify joint connectivity, we compute Euclidean distances and inter-joint angles between key
        # anatomical landmarks such as the hands, shoulders, pelvis, knees, and ankles.

        # Distance key (D1–D8):
        # Pair                                  SMPL joints         Captures
        # D1 L.wrist – L.shoulder               lwrist, lsho        left arm extension (elbow bend)
        # D2 R.wrist – R.shoulder               rwrist, rsho        right arm extension
        # D3 L.shoulder – pelvis                lsho, pelvis        left torso (lateral bend)
        # D4 R.shoulder – pelvis                rsho, pelvis        right torso
        # D5 L.wrist – R.wrist                  lwrist, rwrist      arm span
        # D6 L.shoulder – R.shoulder            lsho, rsho          shoulder width
        # D7 L.ankle – R.ankle                  lankle, rankle      stance width
        # D8 L.knee – R.knee                    lknee, rknee        knee spread (independent in turnout)

        # Angle key (A1–A6):
        # Chain A–B–C                           Vertex B    DOF     Movement
        # A1 L.wrist – L.shoulder – pelvis      lsho        3       arm elevation, abduction, rotation
        # A2 R.wrist – R.shoulder – pelvis      rsho        3       arm elevation, abduction, rotation
        # A3 L.shoulder – pelvis – R.shoulder   pelvis      3       shoulder spread
        # A4 pelvis – L.knee – L.ankle          lknee       1       knee flexion (sagittal)
        # A5 pelvis – R.knee – R.ankle          rknee       1       knee flexion (sagittal)
        # A6 L.knee – pelvis – R.knee           pelvis      3       hip/leg spread

        # --- Vectorized helpers over the whole sequence ---
        def dist_avg_frames(k1, k2):
            a = norm_frames[:, self.IDX[k1], :]
            b = norm_frames[:, self.IDX[k2], :]
            return float(np.mean(np.linalg.norm(a - b, axis=1)))

        def mean_angle(ka, kb, kc):
            """Mean over T frames of the angle (radians) at vertex kb in chain ka-kb-kc."""
            a = norm_frames[:, self.IDX[ka], :]
            b = norm_frames[:, self.IDX[kb], :]
            c = norm_frames[:, self.IDX[kc], :]
            u = a - b
            v = c - b
            nu = np.linalg.norm(u, axis=1)
            nv = np.linalg.norm(v, axis=1)
            valid = (nu > 1e-9) & (nv > 1e-9)
            ang = np.zeros(n_frames)
            if np.any(valid):
                cos = np.sum(u[valid] * v[valid], axis=1) / (nu[valid] * nv[valid])
                ang[valid] = np.arccos(np.clip(cos, -1.0, 1.0))
            return float(np.mean(ang))

        # --- PRE-CALCULATE INITIATION THRESHOLDS (Equation 1 Correction) ---      
        feats = {}

        body_indices = [self.IDX[name] for name in self.BODY_KEY_JOINTS]
        effort_indices = [self.IDX[name] for name in self.EFFORT_KEY_JOINTS]

        # ---------------------------------------------------------
        # BODY COMPONENT (Features 1–23)
        # ---------------------------------------------------------
        # A. Distances D1–D8 (Features 1–8) — doc Eq. 11
        feats["Dist_Hand_Shoulder_L"]   = dist_avg_frames("L_HAND", "L_SHOULDER")     # D1
        feats["Dist_Hand_Shoulder_R"]   = dist_avg_frames("R_HAND", "R_SHOULDER")     # D2
        feats["Dist_Pelvis_Shoulder_L"] = dist_avg_frames("PELVIS", "L_SHOULDER")     # D3
        feats["Dist_Pelvis_Shoulder_R"] = dist_avg_frames("PELVIS", "R_SHOULDER")     # D4
        feats["Dist_Hands"]             = dist_avg_frames("L_HAND", "R_HAND")         # D5
        feats["Dist_Shoulders"]         = dist_avg_frames("L_SHOULDER", "R_SHOULDER") # D6
        feats["Dist_Ankles"]              = dist_avg_frames("L_ANKLE", "R_ANKLE")       # D7
        feats["Dist_Knees"]             = dist_avg_frames("L_KNEE", "R_KNEE")         # D8

        # B. Angles A1–A6 (Features 9–14) — doc Eqs. 12–14
        feats["Angle_LArm"]      = mean_angle("L_WRIST",    "L_SHOULDER", "PELVIS")      # A1
        feats["Angle_RArm"]      = mean_angle("R_WRIST",    "R_SHOULDER", "PELVIS")      # A2
        feats["Angle_Shoulders"] = mean_angle("L_SHOULDER", "PELVIS",     "R_SHOULDER")  # A3
        feats["Angle_LKnee"]     = mean_angle("PELVIS",     "L_KNEE",     "L_ANKLE")     # A4
        feats["Angle_RKnee"]     = mean_angle("PELVIS",     "R_KNEE",     "R_ANKLE")     # A5
        feats["Angle_Hips"]      = mean_angle("L_KNEE",     "PELVIS",     "R_KNEE")      # A6

        # C. Initiation (Features 15–23) — doc Eq. 16:
        #    Init_j = mean over T-w valid frames of 1[s_j(t_i) > tau_j].
        #    raw_init_values is concat(valid_vals[T-w], padded_tail[w]); slice the valid prefix.
        valid_frame_len = n_frames - window_size
        # Curvature (Feature 53) now uses Option-5 asymmetric boundary differences
        # (doc Eqs. 6 & 9), so it no longer needs a 2*window_size span. The binding floor
        # is window_size: the inherently forward Initiation (doc Eq. 7) and backward
        # Effort-Space (doc Eq. 8) sums need at least one valid frame, i.e. n > window_size.
        assert n_frames > window_size, (
            f"Video too short: needs n_frames > {window_size} (window_size), got {n_frames}."
        )

        for joint_name in self.BODY_KEY_JOINTS:
            idx = self.IDX[joint_name]

            # Forward-difference speeds for valid frames i = 1..T-w (doc Eq. 2).
            # Vectorized form of:
            #     velocities = []
            #     for t in range(valid_frame_len):
            #         delta = norm_frames[t + window_size, idx] - norm_frames[t, idx]
            #         velocities.append(np.linalg.norm(delta) / (window_size * self.dt))
            #     velocities = np.array(velocities)
            diffs = norm_frames[window_size:, idx] - norm_frames[:valid_frame_len, idx]
            velocities = np.linalg.norm(diffs, axis=1) / (window_size * self.dt)

            # 2. Threshold = std over the T-w valid values only (doc Eq. 4).
            #    1e-3 floor is a defensive guard against perfectly stationary joints, not in paper.
            sigma = np.std(velocities)
            frame_initiations = np.greater(velocities, sigma)

            feats[f"Initiation_{joint_name}"] = float(np.mean(frame_initiations))

        # =========================================================
        # PER-VIDEO LMA FEATURES — one scalar per sequence, per the reference doc.
        # Build up component by component; each block writes named scalars into `feats`.
        # =========================================================

        # ---------------------------------------------------------
        # EFFORT COMPONENT (Features 24–38) — doc §3
        # NOTE: doc Eqs. 19, 22 specify central differences with lag w/2 (velocity)
        # and lag w (acceleration). `vel`/`acc` above currently use np.gradient
        # (single-frame finite diff). Upgrading those will rescale Weight and Time.
        # ---------------------------------------------------------

        # Doc Eq. 17 — Effort Space_j(T) per joint, then weighted aggregate over J_effort.
        effort_space_total = 0.0
        effort_weight_total = sum([self.weights[joint_name] for joint_name in self.EFFORT_KEY_JOINTS])
        for joint_name in self.EFFORT_KEY_JOINTS:
            idx = self.IDX[joint_name]

            # Backward-difference displacement sum over i = w+1..T (doc Eq. 17, boundary §1.5).
            # Vectorized form of:
            #     total_displacement = 0.0
            #     for t in range(window_size, n_frames):
            #         delta = norm_frames[t, idx] - norm_frames[t - window_size, idx]
            #         total_displacement += np.linalg.norm(delta)
            diffs = norm_frames[window_size:, idx] - norm_frames[:n_frames - window_size, idx]
            total_displacement = float(np.sum(np.linalg.norm(diffs, axis=1)))
            
            net_displacement = np.linalg.norm(norm_frames[n_frames - 1, idx] - norm_frames[0, idx]) + 1e-9
            individual_space = total_displacement / net_displacement

            feats[f"Effort_Space_{joint_name}"] = float(individual_space)
            effort_space_total += self.weights[joint_name] * feats[f"Effort_Space_{joint_name}"]

        feats["Effort_Space_Avg"] = float(effort_space_total) / effort_weight_total      # Feature 24

        # Doc Eqs. 19–21 — Effort Weight: ONE whole-body scalar.
        # Weight = (1/T) Σ_t Σ_{j∈J_effort} ½ α_j v_j(t_i)².
        # Vectorized form of:
        #     w_half = window_size // 2
        #     velocity = np.zeros((n_frames - 2*w_half, 24, 3))
        #     for t in range(n_frames):
        #         for joint_name in self.EFFORT_KEY_JOINTS:
        #             idx = self.IDX[joint_name]
        #             velocity[t] = (norm_frames[t + w_half, idx] - norm_frames[t - w_half, idx]) / (window_size * self.dt)
        #             individual_weight = np.sum(0.5 * self.weights[joint_name] * velocity[t, idx] ** 2)
        #             total_weight += individual_weight
        #     feats["Effort_Weight_Avg"] = total_weight / n_frames

        w_half = window_size // 2
        alphas = np.array([self.weights[name] for name in self.EFFORT_KEY_JOINTS])

        # Central-diff velocity for all 6 effort joints at once. Shape (T - 2*w_half, 6, 3).
        # Denominator uses the actual numerator span (2*w_half) rather than window_size so
        # the v ≈ ΔP/Δt scaling is exact for both odd and even w. Doc-literal (w·τ_f) is
        # off by 1/w for odd w; this matches numerically.
        velocity = (
            norm_frames[2 * w_half : n_frames, effort_indices]   # P(t_i + w_half) for t_i ∈ [w_half, n_frames − w_half)
            - norm_frames[0 : n_frames - 2 * w_half, effort_indices]   # P(t_i − w_half) for the same t_i
        ) / ((2 * w_half) * self.dt)

        # Per-frame whole-body KE: ½ Σ_j α_j ‖v_j‖². Shape (T - 2*w_half,).
        effort_weight_for_all_frames = 0.5 * (np.sum(velocity ** 2, axis=2) @ alphas)

        feats["Effort_Weight_Avg"] = float(np.mean(effort_weight_for_all_frames))                 # Feature 31

        # Doc Eqs. 22-24 — Effort Time: Effort Time_j(T) per joint, then weighted aggregate over J_effort.
        # Time = (1/T) Σ_t Σ_{j∈J_effort} α_j a_j(t_i).
        # Vectorized form of:
        #     w_half = window_size // 2
        #     acceleration = np.zeros((n_frames - 2*w_half, 24, 3))
        #     for t in range(w_half, n_frames - w_half):
        #         for joint_name in self.EFFORT_KEY_JOINTS:
        #             idx = self.IDX[joint_name]
        #             acceleration[t] = (norm_frames[t + w_half, idx] - 2 * norm_frames[t, idx] + norm_frames[t - w_half, idx]) / (w_half * self.dt) ** 2
        #             feats[f"Effort_Time_{joint_name}"] = float(np.mean(acceleration[:, idx]))
        #             total_time += self.weights[joint_name] * feats[f"Effort_Time_{joint_name}"]
        #     feats["Effort_Time"] /= n_frames
        acceleration = (
            norm_frames[2 * w_half:, effort_indices]
            - 2 * norm_frames[w_half:n_frames - w_half, effort_indices]
            + norm_frames[:n_frames - 2 * w_half, effort_indices]
        ) / (w_half * self.dt) ** 2

        # Doc Eq. 23 — Effort Time_j(T) per joint: mean acceleration magnitude over T.
        acc_mag = np.linalg.norm(acceleration, axis=2)  # (T, 24)
        for i, joint_name in enumerate(self.EFFORT_KEY_JOINTS):
            feats[f"Effort_Time_{joint_name}"] = float(np.mean(acc_mag[:, i]))

        # Doc Eq. 24 — Effort Time(T) weighted aggregate over J_effort.
        feats["Effort_Time_Avg"] = float(sum(
            self.weights[joint_name] * feats[f"Effort_Time_{joint_name}"]
            for joint_name in self.EFFORT_KEY_JOINTS
        )) / effort_weight_total                                                                        # Feature 38

        # ---------------------------------------------------------
        # SPACE COMPONENT (Features 39–53 + Feature 55) — doc §4
        # ---------------------------------------------------------

        # Doc §4.1 — Spatial Dispersion: mean ||P_j − P_ref|| over T.
        # Upper body limbs reference SPINE1 (lumbar spine).
        for j in ["L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_HAND", "R_HAND"]:
            feats[f"Disp_Upper_{j}"] = dist_avg_frames(j, "SPINE1")                     # Features 39–44

        # Lower body limbs reference PELVIS.
        for j in ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_FOOT", "R_FOOT"]:
            feats[f"Disp_Lower_{j}"] = dist_avg_frames(j, "PELVIS")                     # Features 45–50

        # Feature 55 (dance-paper only) — Head dispersion from SPINE1.
        feats["Disp_Head"] = dist_avg_frames("HEAD", "SPINE1")

        # Doc §4.2 — Trajectory (pelvis as whole-body proxy).
        pelvis_traj = norm_frames[:, self.IDX["PELVIS"], :]

        # Feature 51 — Total path: sum of single-frame pelvis displacements (doc Eq. 27).
        feats["Traj_Path"] = float(np.sum(np.linalg.norm(pelvis_traj[1:] - pelvis_traj[:-1], axis=1)))

        # Feature 52 — Total distance: net displacement (doc Eq. 28).
        feats["Traj_Distance"] = float(np.linalg.norm(pelvis_traj[-1] - pelvis_traj[0]))

        # Feature 53 — Mean geometric curvature (doc Eqs. 29-32).
        # v (doc Eq. 6, lag w/2) and a (doc Eq. 9, lag w) use OPTION-5 asymmetric boundary
        # differences: forward-looking near the start, backward-looking near the end,
        # centered in the middle, so both are defined on EVERY frame ("no data loss",
        # doc Section 1.5). This is what removes the old n_frames > 2*window_size floor:
        # a clip with n_frames > window_size now yields a curvature.
        pelvis = norm_frames[:, self.IDX["PELVIS"], :]
        w, wh, dt = window_size, w_half, self.dt
        idx = np.arange(n_frames)
        last = n_frames - 1

        # Velocity v(t_i), all scaled by (w*dt): centered in the interior, forward (lag w)
        # near the start, backward (lag w) near the end (doc Eq. 6). Frames whose required
        # neighbour runs off the clip stay NaN and are dropped from the mean.
        vel = np.full((n_frames, 3), np.nan)
        c = (idx >= wh) & (idx <= last - wh)
        vel[c] = (pelvis[idx[c] + wh] - pelvis[idx[c] - wh]) / (w * dt)
        fwd = (idx < wh) & (idx + w <= last)
        vel[fwd] = (pelvis[idx[fwd] + w] - pelvis[idx[fwd]]) / (w * dt)
        bwd = (idx > last - wh) & (idx - w >= 0)
        vel[bwd] = (pelvis[idx[bwd]] - pelvis[idx[bwd] - w]) / (w * dt)

        # Acceleration a(t_i): centered (lag w) where it fits, else derived from the
        # velocity field with lag w/2 at the boundaries (doc Eq. 9). Fill order
        # centered -> forward -> backward never overwrites a finite value with NaN.
        acc = np.full((n_frames, 3), np.nan)
        cA = (idx >= w) & (idx <= last - w)
        acc[cA] = (pelvis[idx[cA] + w] - 2 * pelvis[idx[cA]] + pelvis[idx[cA] - w]) / (w * dt) ** 2

        def _fill(mask, vals):
            take = mask & ~np.isfinite(acc).all(axis=1)
            acc[take] = vals[take]

        if wh > 0:
            tmp = np.full((n_frames, 3), np.nan)
            fA = (idx < w) & (idx + wh <= last)
            tmp[fA] = (vel[idx[fA] + wh] - vel[idx[fA]]) / (wh * dt)
            _fill(fA, tmp)
            tmp = np.full((n_frames, 3), np.nan)
            bA = (idx > last - w) & (idx - wh >= 0)
            tmp[bA] = (vel[idx[bA]] - vel[idx[bA] - wh]) / (wh * dt)
            _fill(bA, tmp)

        # kappa = |v x a| / |v|^3, averaged over frames where v, a are finite and |v| > 0.
        good = np.isfinite(vel).all(axis=1) & np.isfinite(acc).all(axis=1)
        v_norm = np.linalg.norm(np.where(good[:, None], vel, 0.0), axis=1)
        good &= v_norm ** 3 > 1e-12
        if np.any(good):
            kappa = np.linalg.norm(np.cross(vel[good], acc[good]), axis=1) / (v_norm[good] ** 3)
            feats["Traj_Curvature_Avg"] = float(np.mean(kappa))
        else:
            feats["Traj_Curvature_Avg"] = 0.0

        # ---------------------------------------------------------
        # SHAPE COMPONENT (Feature 54) — doc §5
        # ConvexHull volume over all 24 SMPL JOINT positions per frame, averaged over T.
        # Note: doc specifies joints (J_all = 24), not mesh vertices. The `all_volumes`
        # arg coming from process_lma_features.py is mesh-based — intentionally unused here.
        # ---------------------------------------------------------
        vol_all_frames = np.zeros(n_frames)
        for i in range(n_frames):
            try:
                vol_all_frames[i] = ConvexHull(norm_frames[i]).volume
            except Exception:
                # Degenerate frame (collinear/coplanar joints, NaNs): leave at 0.0
                vol_all_frames[i] = 0.0
        feats["Body_Volume_Avg"] = float(np.mean(vol_all_frames))                      # Feature 54

        return feats

    def extract_per_frame_features(self, all_frames, all_floor_models):
        """Per-FRAME (T,55) LMA stream — the same 55 features evaluated at EVERY frame
        using lag w, BEFORE the time-aggregation that extract_all_features applies.
        Returns a dict {name: (T,) array} with the SAME keys as extract_all_features, so
        np.stack([pf[k] for k in sorted(pf)], axis=1) is a (T,55) whose columns match the
        (1,55). For time-MEAN features the column mean equals the aggregate scalar (this is
        the correctness oracle); sum/ratio/net features (Effort_Space, Traj_Path/Distance,
        Initiation) use their natural per-frame decomposition. Dynamic lag-w terms are
        edge-padded at the boundary frames so every frame is defined. Same preprocessing
        (impute + floor-normalize) and the corrected SMPL-24 joints as extract_all_features.
        """
        cleaned = self._impute_missing_data(all_frames)
        norm = self._normalize_pose_to_floor(cleaned, all_floor_models)
        T = len(all_frames)
        w = self.window_size
        wh = w // 2
        dt = self.dt
        assert T > w, f"need T > window_size ({w}), got {T}"
        pf = {}

        def align(vals, start):
            """Place a shorter valid array at [start, start+len) and edge-pad to length T."""
            out = np.empty(T, dtype=float)
            n = len(vals)
            out[start:start + n] = vals
            if start > 0:
                out[:start] = vals[0]
            if start + n < T:
                out[start + n:] = vals[-1]
            return out

        def dist_pf(k1, k2):
            a = norm[:, self.IDX[k1], :]; b = norm[:, self.IDX[k2], :]
            return np.linalg.norm(a - b, axis=1)

        def angle_pf(ka, kb, kc):
            a = norm[:, self.IDX[ka]]; b = norm[:, self.IDX[kb]]; c = norm[:, self.IDX[kc]]
            u = a - b; v = c - b
            nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
            out = np.zeros(T); valid = (nu > 1e-9) & (nv > 1e-9)
            cos = np.sum(u[valid] * v[valid], axis=1) / (nu[valid] * nv[valid])
            out[valid] = np.arccos(np.clip(cos, -1.0, 1.0))
            return out

        # BODY distances D1-D8 (exact per-frame; mean == aggregate)
        pf["Dist_Hand_Shoulder_L"]   = dist_pf("L_HAND", "L_SHOULDER")
        pf["Dist_Hand_Shoulder_R"]   = dist_pf("R_HAND", "R_SHOULDER")
        pf["Dist_Pelvis_Shoulder_L"] = dist_pf("PELVIS", "L_SHOULDER")
        pf["Dist_Pelvis_Shoulder_R"] = dist_pf("PELVIS", "R_SHOULDER")
        pf["Dist_Hands"]             = dist_pf("L_HAND", "R_HAND")
        pf["Dist_Shoulders"]         = dist_pf("L_SHOULDER", "R_SHOULDER")
        pf["Dist_Ankles"]            = dist_pf("L_ANKLE", "R_ANKLE")
        pf["Dist_Knees"]             = dist_pf("L_KNEE", "R_KNEE")
        # BODY angles A1-A6 (exact per-frame; mean == aggregate)
        pf["Angle_LArm"]      = angle_pf("L_WRIST", "L_SHOULDER", "PELVIS")
        pf["Angle_RArm"]      = angle_pf("R_WRIST", "R_SHOULDER", "PELVIS")
        pf["Angle_Shoulders"] = angle_pf("L_SHOULDER", "PELVIS", "R_SHOULDER")
        pf["Angle_LKnee"]     = angle_pf("PELVIS", "L_KNEE", "L_ANKLE")
        pf["Angle_RKnee"]     = angle_pf("PELVIS", "R_KNEE", "R_ANKLE")
        pf["Angle_Hips"]      = angle_pf("L_KNEE", "PELVIS", "R_KNEE")
        # Initiation 15-23: per-frame indicator 1[v(t) > sigma] (forward lag-w speed; mean == aggregate)
        for jn in self.BODY_KEY_JOINTS:
            idx = self.IDX[jn]
            diffs = norm[w:, idx] - norm[:T - w, idx]
            vel = np.linalg.norm(diffs, axis=1) / (w * dt)
            pf[f"Initiation_{jn}"] = align((vel > np.std(vel)).astype(float), 0)
        # EFFORT Space 24-31: per-frame lag-w displacement / net (sum == aggregate)
        eff_w_total = sum(self.weights[j] for j in self.EFFORT_KEY_JOINTS)
        es_avg = np.zeros(T)
        for jn in self.EFFORT_KEY_JOINTS:
            idx = self.IDX[jn]
            disp = np.linalg.norm(norm[w:, idx] - norm[:T - w, idx], axis=1)
            net = np.linalg.norm(norm[T - 1, idx] - norm[0, idx]) + 1e-9
            es = align(disp / net, 0)
            pf[f"Effort_Space_{jn}"] = es
            es_avg += self.weights[jn] * es
        pf["Effort_Space_Avg"] = es_avg / eff_w_total
        # EFFORT Weight 31: per-frame whole-body KE (central lag w_half; mean ~ aggregate)
        eidx = [self.IDX[j] for j in self.EFFORT_KEY_JOINTS]
        alphas = np.array([self.weights[j] for j in self.EFFORT_KEY_JOINTS])
        vel_c = (norm[2 * wh:T, eidx] - norm[0:T - 2 * wh, eidx]) / ((2 * wh) * dt)
        pf["Effort_Weight_Avg"] = align(0.5 * (np.sum(vel_c ** 2, axis=2) @ alphas), wh)
        # EFFORT Time 32-38: per-frame acceleration magnitude per joint (central lag w_half; mean ~ aggregate)
        acc_c = (norm[2 * wh:, eidx] - 2 * norm[wh:T - wh, eidx] + norm[:T - 2 * wh, eidx]) / (wh * dt) ** 2
        acc_mag = np.linalg.norm(acc_c, axis=2)
        for i, jn in enumerate(self.EFFORT_KEY_JOINTS):
            pf[f"Effort_Time_{jn}"] = align(acc_mag[:, i], wh)
        pf["Effort_Time_Avg"] = sum(self.weights[jn] * pf[f"Effort_Time_{jn}"]
                                    for jn in self.EFFORT_KEY_JOINTS) / eff_w_total
        # SPACE dispersions 39-50 + 55 (exact per-frame; mean == aggregate)
        for j in ["L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_HAND", "R_HAND"]:
            pf[f"Disp_Upper_{j}"] = dist_pf(j, "SPINE1")
        for j in ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_FOOT", "R_FOOT"]:
            pf[f"Disp_Lower_{j}"] = dist_pf(j, "PELVIS")
        pf["Disp_Head"] = dist_pf("HEAD", "SPINE1")
        # SPACE trajectory 51-52: per-frame pelvis step (sum == path); running net displacement (last == distance)
        pelvis = norm[:, self.IDX["PELVIS"], :]
        pf["Traj_Path"] = align(np.linalg.norm(pelvis[1:] - pelvis[:-1], axis=1), 1)
        pf["Traj_Distance"] = np.linalg.norm(pelvis - pelvis[0], axis=1)
        # SPACE curvature 53: per-frame kappa (Option-5 vel/acc, same as aggregate; 0 where undefined)
        idxa = np.arange(T); last = T - 1
        vel = np.full((T, 3), np.nan)
        c = (idxa >= wh) & (idxa <= last - wh);  vel[c]   = (pelvis[idxa[c] + wh] - pelvis[idxa[c] - wh]) / (w * dt)
        fwd = (idxa < wh) & (idxa + w <= last);  vel[fwd] = (pelvis[idxa[fwd] + w] - pelvis[idxa[fwd]]) / (w * dt)
        bwd = (idxa > last - wh) & (idxa - w >= 0); vel[bwd] = (pelvis[idxa[bwd]] - pelvis[idxa[bwd] - w]) / (w * dt)
        acc = np.full((T, 3), np.nan)
        cA = (idxa >= w) & (idxa <= last - w); acc[cA] = (pelvis[idxa[cA] + w] - 2 * pelvis[idxa[cA]] + pelvis[idxa[cA] - w]) / (w * dt) ** 2

        def _fill(mask, vals):
            take = mask & ~np.isfinite(acc).all(axis=1)
            acc[take] = vals[take]
        if wh > 0:
            tmp = np.full((T, 3), np.nan); fA = (idxa < w) & (idxa + wh <= last)
            tmp[fA] = (vel[idxa[fA] + wh] - vel[idxa[fA]]) / (wh * dt); _fill(fA, tmp)
            tmp = np.full((T, 3), np.nan); bA = (idxa > last - w) & (idxa - wh >= 0)
            tmp[bA] = (vel[idxa[bA]] - vel[idxa[bA] - wh]) / (wh * dt); _fill(bA, tmp)
        good = np.isfinite(vel).all(axis=1) & np.isfinite(acc).all(axis=1)
        vn = np.linalg.norm(np.where(good[:, None], vel, 0.0), axis=1)
        good &= vn ** 3 > 1e-12
        kappa = np.zeros(T)
        kappa[good] = np.linalg.norm(np.cross(vel[good], acc[good]), axis=1) / (vn[good] ** 3)
        pf["Traj_Curvature_Avg"] = kappa
        # SHAPE 54: per-frame ConvexHull volume (exact per-frame; mean == aggregate)
        vol = np.zeros(T)
        for i in range(T):
            try:
                vol[i] = ConvexHull(norm[i]).volume
            except Exception:
                vol[i] = 0.0
        pf["Body_Volume_Avg"] = vol
        return pf

    def _add_feat(self, feat_dict, key, val, t):
        if key not in feat_dict:
            # Use body_volume as the length reference
            feat_dict[key] = np.zeros(len(feat_dict["body_volume"]))
        feat_dict[key][t] = val