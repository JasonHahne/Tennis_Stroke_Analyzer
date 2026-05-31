import numpy as np
from dtaidistance import dtw


class PoseComparator:
    def normalize_pose(self, landmarks_3d):
        out = landmarks_3d.copy()
        for i in range(len(out)):
            hip_mid = (out[i, 23] + out[i, 24]) / 2.0
            out[i] -= hip_mid
            shoulder_w = np.linalg.norm(out[i, 11] - out[i, 12])
            if shoulder_w > 1e-6:
                out[i] /= shoulder_w
        return out

    @staticmethod
    def _resample(arr, n):
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, arr)

    def compute_joint_errors(self, user_angles, pro_angles):
        errors = {}
        for joint in user_angles:
            if joint not in pro_angles:
                continue
            u = user_angles[joint].astype(np.double)
            p = pro_angles[joint].astype(np.double)
            n = min(len(u), len(p), 300)
            u_r = self._resample(u, n)
            p_r = self._resample(p, n)
            dtw_dist = dtw.distance_fast(u_r, p_r)
            errors[joint] = {
                'dtw_dist':   float(dtw_dist),
                'mean_error': float(np.mean(np.abs(u_r - p_r))),
                'max_error':  float(np.max(np.abs(u_r - p_r))),
                'user_mean':  float(np.mean(u)),
                'pro_mean':   float(np.mean(p)),
                'user_arr':   u_r,
                'pro_arr':    p_r,
            }
        return errors

    def generate_feedback(self, joint_errors, stroke_type):
        THRESHOLD = 12
        tips = {
            'right_elbow': (
                f"Extend your right elbow more at contact on the {stroke_type}.",
                f"Keep your right elbow more bent; don't over-extend on the {stroke_type}.",
            ),
            'left_elbow': (
                f"Your left (non-dominant) elbow is too open. Tuck it slightly during the {stroke_type}.",
                f"Your left elbow is too bent. Let it extend for better balance on the {stroke_type}.",
            ),
            'right_shoulder': (
                f"Increase shoulder rotation on your {stroke_type} to generate more power.",
                f"Reduce shoulder over-rotation on the {stroke_type}; it disrupts timing.",
            ),
            'left_shoulder': (
                f"Drop your left shoulder less; keep it level during the {stroke_type}.",
                f"Raise your left shoulder slightly for better trunk stability on the {stroke_type}.",
            ),
            'hip_rotation': (
                f"Add more hip rotation — drive from the legs on your {stroke_type}.",
                f"Your hips over-rotate. Control the hip turn to improve {stroke_type} accuracy.",
            ),
            'right_knee': (
                f"Bend your right knee more; stay low through the {stroke_type}.",
                f"Your right knee is over-bent — stand up slightly into the {stroke_type}.",
            ),
        }
        feedback = []
        for joint, errs in sorted(joint_errors.items(), key=lambda x: -x[1]['mean_error']):
            if errs['mean_error'] > THRESHOLD and joint in tips:
                idx = 0 if errs['user_mean'] < errs['pro_mean'] else 1
                feedback.append(tips[joint][idx])

        if not feedback:
            feedback.append(
                f"Great {stroke_type}! Your technique closely matches professional standards."
            )
        return feedback