import os
import platform
import ctypes
import cv2
import numpy as np

YOLO = None
YOLO_IMPORT_ERROR = None

try:
    if platform.system() == "Windows":
        try:
            from importlib.util import find_spec
            spec = find_spec("torch")
            if spec and spec.origin:
                dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
                if os.path.exists(dll_path):
                    ctypes.CDLL(os.path.normpath(dll_path))
        except Exception:
            pass

    from ultralytics import YOLO
except Exception as e:
    YOLO_IMPORT_ERROR = e
    YOLO = None


class RacketTracker:
    RACKET_CLASS = 38  # COCO class index for tennis racket

    def __init__(self):
        self.available = YOLO is not None
        self.model = None

        if self.available:
            try:
                self.model = YOLO('yolov8n.pt')
            except Exception as e:
                self.available = False
                self.model = None
                self.init_error = e
        else:
            self.init_error = YOLO_IMPORT_ERROR

    def detect(self, frame):
        if not self.available or self.model is None:
            return None

        results = self.model(frame, verbose=False)
        best = None
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != self.RACKET_CLASS:
                    continue
                conf = float(box.conf[0])
                if conf < 0.25:
                    continue
                if best is None or conf > best['conf']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    angle = float(np.degrees(np.arctan2(h, max(w, 1))))
                    best = dict(
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        width=w,
                        height=h,
                        angle=angle,
                        conf=conf
                    )
        return best

    def track_video(self, video_path, callback=None):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data = []
        idx = 0

        if not self.available or self.model is None:
            while cap.isOpened():
                ret, _frame = cap.read()
                if not ret:
                    break
                data.append(None)
                if callback:
                    callback(idx, total)
                idx += 1
            cap.release()
            return data

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            data.append(self.detect(frame))
            if callback:
                callback(idx, total)
            idx += 1

        cap.release()
        return data

    @staticmethod
    def draw_racket_2d(frame, det, color=(0, 255, 255), thickness=2):
        if det is None:
            return frame

        cx, cy = det['center']
        w, h = max(det['width'] // 2, 20), max(det['height'] // 2, 15)
        angle = det['angle']

        cv2.ellipse(
            frame,
            (cx, cy - det['height'] // 4),
            (w, h),
            angle,
            0,
            360,
            color,
            thickness
        )

        rad = np.radians(angle + 90)
        hlen = max(det['height'] // 2, 30)
        hx = int(cx + hlen * np.cos(rad))
        hy = int(cy + det['height'] // 4 + hlen * np.sin(rad))

        cv2.line(
            frame,
            (cx, cy + det['height'] // 4),
            (hx, hy),
            color,
            thickness + 1
        )
        return frame
