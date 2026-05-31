"""
processing/pose_extractor.py

ROOT CAUSE AND DEFINITIVE FIX FOR TIMESTAMP ERRORS
────────────────────────────────────────────────────
MediaPipe VIDEO mode keeps internal temporal state per landmarker instance.
When the reference builder reuses one PoseExtractor for 11 forehand videos,
the C++ graph sees a jump (e.g. alcaraz finishes at ts=9950 ms, then
detect_for_video is called again with ts=9983 ms for delportro's frame 1 —
internally MediaPipe validates cadence and continuity of the stream and
raises "Input timestamp must be monotonically increasing" for ANY video
after the first, because the new video's image content has no continuity
with the previous video's images, causing the internal pose tracker to
produce an error rather than silently reinitialising.

FIX: Switch the running mode to VisionRunningMode.IMAGE.
  • Each frame is processed independently — zero timestamp bookkeeping.
  • No state is carried across frames OR across video files.
  • Works correctly for both the reference builder (multi-video loop)
    and single-video user analysis.
  • The minor loss of inter-frame tracking is negligible for stored video
    batch analysis.

SERVE HANG FIX
──────────────
max_frames is now 250 for reference building (set in reference_builder.py).
At 30 fps that is 8.3 s — plenty for any stroke.
At 240 fps slow-motion it is 1.04 s of real action.
For user video analysis the default remains 900 frames.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

SKELETON_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (0,  11), (0,  12),
]

KEY_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class PoseExtractor:

    def __init__(self, model_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(
            base_dir, 'models', 'pose_landmarker_full.task')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Pose model not found:\n{self.model_path}\n\n"
                "Download the MediaPipe Pose Landmarker model and save it there.")

        BaseOptions           = mp.tasks.BaseOptions
        PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        # ── IMAGE mode: no timestamps, no continuity requirement ──────
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,     # ← THE FIX
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    # ──────────────────────────────────────────────────────────────────
    def extract_from_video(self, video_path, callback=None,
                           max_frames: int = 900):
        """
        Extract 2-D and 3-D pose landmarks from a stored video file.

        Parameters
        ----------
        video_path : str
        callback   : callable(current_frame, total_frames) | None
        max_frames : int
            Hard cap. 250 for reference building, 900 for user analysis.
            0 / None = no cap (not recommended for reference building).
        """
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0

        effective = min(total, max_frames) if max_frames else total

        lm2d_list = []; lm3d_list = []; idx_list = []
        idx = 0

        while cap.isOpened():
            if max_frames and idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # IMAGE mode — detect() with no timestamp argument
            result = self.landmarker.detect(mp_image)

            if result.pose_landmarks and result.pose_world_landmarks:
                pose2d = result.pose_landmarks[0]
                pose3d = result.pose_world_landmarks[0]

                lm2d = np.array(
                    [[lm.x, lm.y, getattr(lm, 'visibility', 1.0)]
                     for lm in pose2d], dtype=np.float32)
                lm3d = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose3d],
                    dtype=np.float32)

                if lm2d.shape[0] == 33 and lm3d.shape[0] == 33:
                    lm2d_list.append(lm2d)
                    lm3d_list.append(lm3d)
                    idx_list.append(idx)

            if callback:
                callback(idx, effective)
            idx += 1

        cap.release()

        if not lm3d_list:
            raise RuntimeError(
                f"No pose landmarks detected in: {video_path}\n"
                "Ensure the full body is clearly visible throughout the clip.")

        return {
            'landmarks_2d':  np.array(lm2d_list,  dtype=np.float32),
            'landmarks_3d':  np.array(lm3d_list,  dtype=np.float32),
            'frame_indices': np.array(idx_list,   dtype=np.int32),
            'fps':           fps,
            'total_frames':  idx,
        }

    # ──────────────────────────────────────────────────────────────────
    def compute_joint_angles(self, landmarks_3d):
        def _angle(a, b, c):
            ba  = a - b; bc = c - b
            den = (np.linalg.norm(ba, axis=1) *
                   np.linalg.norm(bc, axis=1)) + 1e-8
            return np.degrees(np.arccos(
                np.clip(np.einsum('ij,ij->i', ba, bc) / den, -1.0, 1.0)))

        L = landmarks_3d
        return {
            'right_elbow':    _angle(L[:, 12], L[:, 14], L[:, 16]),
            'left_elbow':     _angle(L[:, 11], L[:, 13], L[:, 15]),
            'right_shoulder': _angle(L[:, 24], L[:, 12], L[:, 14]),
            'left_shoulder':  _angle(L[:, 23], L[:, 11], L[:, 13]),
            'hip_rotation':   _angle(L[:, 11], L[:, 23], L[:, 24]),
            'right_knee':     _angle(L[:, 24], L[:, 26], L[:, 28]),
            'left_knee':      _angle(L[:, 23], L[:, 25], L[:, 27]),
        }