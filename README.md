# Tennis Stroke Analyzer

The Tennis Stroke Analyzer is a computer vision application built to help players evaluate and improve their technique. By utilizing deep learning models, the software tracks human pose landmarks and tennis rackets from video recordings, extracting 3D joint angles to deliver a side-by-side comparative analysis against reference data compiled from professional players.

## Key Features

* **Biomechanical Tracking:** Leverages MediaPipe to capture precise 3D body coordinates independent of camera distance.
* **Equipment Detection:** Integrates YOLOv8 object detection to track the path, center, and angle of the tennis racket throughout the movement.
* **Automated Stroke Identification:** Supports data collection and profile generation for major strokes: Forehands, Backhands, and Serves.
* **Sequence Alignment:** Utilizes Dynamic Time Warping (DTW) to dynamically map user movements to professional templates, correcting for differences in timing and speed.
* **Graphical User Interface:** Built with PyQt5, featuring live overlay visualizers, interactive charts, and structured feedback panels.

## Visual Demonstrations

### 1. 2D Pose and Racket Tracking View
This view displays the video analysis with the 2D skeleton overlay and real-time tennis racket tracking boundary graphics.

![2D Tracking Demonstration](images/2d_overlay.gif)

### 2. 3D Joint Reconstruction View
This panel maps out the normalized 3D coordinate system, isolating human joint positioning relative to professional archetypes.

![3D Joint Reconstruction Demonstration](images/3d_view.gif)

### 3. Ball Trajectory and Motion Analysis
This visualization plots the exact path, velocity metrics, and launch angles calculated from the sequence data.

![Ball Trajectory Plot Demonstration](images/balltrajectory.gif)

## Architecture and Pipeline

1. **Pose and Angle Extraction:** Video frames are individually processed to avoid sequence-dependent tracking errors, computing exact angles for shoulders, elbows, hips, and knees.
2. **Professional Reference Modeling:** Aggregates multi-video samples of professional strokes to calculate a standardized biomechanical standard.
3. **Similarity Analysis:** Resamples and aligns timelines via DTW to pinpoint exactly where user geometry diverges from professional standards.
4. **Feedback Generation:** Computes joint error thresholds to output targeted text suggestions for technical improvement.

## Installation and Setup

### Prerequisites
* Python 3.8 or higher
* NVIDIA GPU (Optional, recommended for faster YOLO inference)

### Dependencies
Install the required third-party libraries:

```bash
pip install numpy opencv-python PyQt5 matplotlib ultralytics dtaidistance