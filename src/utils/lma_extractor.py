import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter

class LMAExtractor:
    def __init__(self, window_size=55, fps=60):
        """
        Laban Movement Analysis Feature Extractor.
        Faithfully implements the 55-feature vector described in Turab et al. (2025),
        incorporating specific lag-based Space metrics and threshold-based Initiation.
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # Standard SMPL 24-joint topology
        self.IDX = {
            "PELVIS": 0,
            "L_HIP": 1,
            "R_HIP": 2,
            "SPINE1": 3,
            "L_KNEE": 4,
            "R_KNEE": 5,
            "SPINE2": 6,
            "L_ANKLE": 7,
            "R_ANKLE": 8,
            "SPINE3": 9,
            "L_FOOT": 10,
            "R_FOOT": 11,
            "NECK": 12,
            "L_COLLAR": 13,
            "R_COLLAR": 14,
            "HEAD": 15,
            "L_SHOULDER": 16,
            "R_SHOULDER": 17,
            "L_ELBOW": 18,
            "R_ELBOW": 19,
            "L_WRIST": 20,
            "R_WRIST": 21,
            "L_HAND": 22,
            "R_HAND": 23,
        }

        # [cite_start]The 6 Key Joints identified from SHAP plots and Effort descriptions [cite: 183-191]
        self.KEY_JOINTS = ["HEAD", "PELVIS", "L_WRIST", "R_WRIST", "L_ANKLE", "R_ANKLE"]

        # Weights for Global Sums (extremities get higher weight)
        self.weights = {k: 1.0 for k in self.KEY_JOINTS}
        for k in ["L_WRIST", "R_WRIST", "L_ANKLE", "R_ANKLE"]:
            self.weights[k] = 1.5

    def _impute_missing_data(self, joint_seq):
        """Linearly interpolates missing frames to ensure continuity."""
        n_frames = len(joint_seq)
        valid_indices = [i for i, x in enumerate(joint_seq) if len(x) > 0]

        if not valid_indices:
            return np.zeros((n_frames, 24, 3))

        full_seq = np.zeros((n_frames, 24, 3))

        # Fill known values
        for i in valid_indices:
            full_seq[i] = joint_seq[i]

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

    def extract_all_features(self, all_joints, all_volumes, all_floor_models):
        """
        Extracts the 55 LMA features with corrected Equation 1 & 2 logic.
        """
        # 1. Preprocessing
        cleaned_joints = self._impute_missing_data(all_joints)
        norm_joints = self._normalize_pose_to_floor(cleaned_joints, all_floor_models)

        # Applies a temporal median filter (kernel=5) to smooth 3D pose jitter
        # This brings acceleration from ~500 m/s^2 down to realistic human levels.
        window_len = int(self.fps / 4)  # ~0.25s at 60fps
        poly_order = 3

        # 1. Smooth Positions (for Space/Shape/Body)
        for j in range(24):
            for c in range(3):
                norm_joints[:, j, c] = savgol_filter(norm_joints[:, j, c], window_len, poly_order, deriv=0)

        # 2. Calculate Derivatives (for Effort) directly from the filter
        # This is much cleaner than np.gradient(np.gradient(...))
        vel = np.zeros_like(norm_joints)
        acc = np.zeros_like(norm_joints)
        jerk = np.zeros_like(norm_joints)

        for j in range(24):
            for c in range(3):
                vel[:, j, c] = savgol_filter(norm_joints[:, j, c], window_len, poly_order, deriv=1, delta=self.dt)
                acc[:, j, c] = savgol_filter(norm_joints[:, j, c], window_len, poly_order, deriv=2, delta=self.dt)
                jerk[:, j, c] = savgol_filter(norm_joints[:, j, c], window_len, poly_order, deriv=3, delta=self.dt)

        n_frames = len(all_joints)
        w_main = self.window_size

        # --- PRE-CALCULATE INITIATION THRESHOLDS (Equation 1 Correction) ---
        # [cite_start]"Threshold calculated using standard-deviation of the entire sequence" [cite: 98]
        # We calculate the raw initiation metric for the whole video first.
        w_init = 5  # Short window for initiation
        init_thresholds = {}
        raw_init_values = {}

        for name in self.KEY_JOINTS:
            idx = self.IDX[name]
            raw_vals = []
            
            # 1. Calculate for full windows
            for t in range(n_frames - w_init):
                # Raw metric: ||P(t+w) - P(t)|| / dt
                delta = norm_joints[t + w_init, idx] - norm_joints[t, idx]
                val = np.linalg.norm(delta) / (w_init * self.dt)
                raw_vals.append(val)
            
            # 2. Handle the edge case (last w_init frames) by repeating last valid value
            if raw_vals:
                last_val = raw_vals[-1]
                for _ in range(w_init):
                    raw_vals.append(last_val)
            else:
                # Fallback for videos shorter than w_init
                raw_vals = [0.0] * n_frames

            raw_vals = np.array(raw_vals)
            
            # 3. Compute Thresholds
            if len(raw_vals) > 0:
                sigma = np.std(raw_vals)
                init_thresholds[name] = max(sigma, 1e-3)
                raw_init_values[name] = raw_vals # Now full length, no padding needed
            else:
                init_thresholds[name] = 1.0
                raw_init_values[name] = np.zeros(n_frames)

        # Initialize Dictionary
        feats = {"body_volume": np.zeros(n_frames)}

        # 3. Frame-by-Frame Extraction
        for t in range(n_frames):
            # Causal Window: [t-w+1 : t+1]
            start = max(0, t - w_main + 1)
            end = t + 1
            curr_pose = norm_joints[t]

            # ---------------------------------------------------------
            # COMPONENT 1: RAW KINEMATICS (6 Features)
            #
            # ---------------------------------------------------------
            for name in self.KEY_JOINTS:
                idx = self.IDX[name]
                v_mag = np.mean(np.linalg.norm(vel[start:end, idx, :], axis=1))
                self._add_feat(feats, f"{name}_vel", v_mag, t)

            # ---------------------------------------------------------
            # COMPONENT 2: EFFORT (28 Features)
            # ---------------------------------------------------------
            sums = {"Weight": 0, "Time": 0, "Flow": 0, "Space": 0}

            for name in self.KEY_JOINTS:
                idx = self.IDX[name]
                wt = self.weights[name]

                # A. Weight (Kinetic Energy) [Eq 4]
                v_sq = np.sum(vel[start:end, idx, :] ** 2, axis=1)
                ke = np.mean(0.5 * v_sq)
                self._add_feat(feats, f"{name}_KE", ke, t)
                sums["Weight"] += ke * wt

                # B. Time (Acceleration) [Eq 5]
                a_mag = np.mean(np.linalg.norm(acc[start:end, idx, :], axis=1))
                self._add_feat(feats, f"{name}_Accel", a_mag, t)
                sums["Time"] += a_mag * wt

                # C. Flow (Jerkiness)
                j_mag = np.mean(np.linalg.norm(jerk[start:end, idx, :], axis=1))
                self._add_feat(feats, f"{name}_Jerk", j_mag, t)
                sums["Flow"] += j_mag * wt

                # D. Space (Lagged Directness) [Eq 2 Correction]
                # [cite_start]"Sum of distances... ||P(t) - P(t-w)||... for a short time-window" [cite: 104-105]
                # We interpret 'w' as a lag parameter distinct from window T.
                # If w > 1, this measures volumetric "reach" rather than arc length.
                w_lag = 5  # Lag of ~0.08s (5 frames @ 60fps)

                traj = norm_joints[start:end, idx, :]  # Shape (Window_Len, 3)
                win_len = len(traj)

                numerator = 0.0
                if win_len > w_lag:
                    # Sum ||P(i) - P(i-w)||
                    for i in range(w_lag, win_len):
                        dist_lag = np.linalg.norm(traj[i] - traj[i - w_lag])
                        numerator += dist_lag
                else:
                    # Fallback for very first frames
                    numerator = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

                # Denominator: ||P(T) - P(t1)|| (Displacement of the whole window)
                disp = np.linalg.norm(traj[-1] - traj[0])

                # Faithfully implementing Eq 2 without log-scaling for Random Forest
                space_val = numerator / (disp + 1e-6)
                if space_val < 1.0:
                    space_val = 1.0

                self._add_feat(feats, f"{name}_Directness", space_val, t)
                sums["Space"] += space_val * wt

            self._add_feat(feats, "Effort_Weight_Global", sums["Weight"], t)
            self._add_feat(feats, "Effort_Time_Global", sums["Time"], t)
            self._add_feat(feats, "Effort_Flow_Global", sums["Flow"], t)
            self._add_feat(feats, "Effort_Space_Global", sums["Space"], t)

            # ---------------------------------------------------------
            # COMPONENT 3: SPACE (8 Features)
            # ---------------------------------------------------------
            def dist(k1, k2):
                return np.linalg.norm(curr_pose[self.IDX[k1]] - curr_pose[self.IDX[k2]])

            self._add_feat(feats, "Dispersion_Head", dist("HEAD", "SPINE2"), t)
            self._add_feat(feats, "Dispersion_R_Wrist", dist("R_WRIST", "SPINE2"), t)
            self._add_feat(feats, "Dispersion_L_Wrist", dist("L_WRIST", "SPINE2"), t)
            self._add_feat(feats, "Dispersion_R_Ankle", dist("R_ANKLE", "PELVIS"), t)
            self._add_feat(feats, "Dispersion_L_Ankle", dist("L_ANKLE", "PELVIS"), t)

            root_traj = norm_joints[start:end, self.IDX["PELVIS"], :]
            total_path = np.sum(np.linalg.norm(np.diff(root_traj, axis=0), axis=1))
            total_disp = np.linalg.norm(root_traj[-1] - root_traj[0])

            # Clamp to 1.0 to satisfy Triangle Inequality (Path >= Displacement)
            val = total_path / (total_disp + 1e-6)
            curvature = max(1.0, val)

            self._add_feat(feats, "Traj_Path_Length", total_path, t)
            self._add_feat(feats, "Traj_Displacement", total_disp, t)
            self._add_feat(feats, "Traj_Curvature", curvature, t)

            # ---------------------------------------------------------
            # COMPONENT 4: SHAPE (1 Feature)
            # ---------------------------------------------------------
            feats["body_volume"][t] = all_volumes[t]

            # ---------------------------------------------------------
            # COMPONENT 5: BODY (12 Features)
            # ---------------------------------------------------------
            # A. Distances
            self._add_feat(feats, "Dist_Hand_Shoulder_L", dist("L_WRIST", "L_SHOULDER"), t)
            self._add_feat(feats, "Dist_Hand_Shoulder_R", dist("R_WRIST", "R_SHOULDER"), t)
            self._add_feat(feats, "Dist_Ankle_Knee_L", dist("L_ANKLE", "L_KNEE"), t)
            self._add_feat(feats, "Dist_Ankle_Knee_R", dist("R_ANKLE", "R_KNEE"), t)
            self._add_feat(feats, "Dist_Hands", dist("L_WRIST", "R_WRIST"), t)
            self._add_feat(feats, "Dist_Feet", dist("L_ANKLE", "R_ANKLE"), t)

            # B. Initiation (Eq 1 Correction)
            # [cite_start]"Initiation(t) ... > epsilon" [cite: 96]
            # Implements the detection event logic rather than continuous value.
            for name in self.KEY_JOINTS:
                raw_val = raw_init_values[name][t]
                threshold = init_thresholds[name]

                # Boolean Thresholding (Detection Event)
                if raw_val > threshold:
                    init_feat = 1.0
                else:
                    init_feat = 0.0

                self._add_feat(feats, f"Initiation_{name}", init_feat, t)

        # Remove valid-but-duplicate initialization keys to return exactly 55 features
        for duplicate in ["weight", "time", "flow", "space"]:
            if duplicate in feats:
                del feats[duplicate]
        
        return feats

    def _add_feat(self, feat_dict, key, val, t):
        if key not in feat_dict:
            # Use body_volume as the length reference
            feat_dict[key] = np.zeros(len(feat_dict["body_volume"]))
        feat_dict[key][t] = val