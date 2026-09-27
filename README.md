# Xeptocore: Real-Time Deepfake Intrusion Detection System

Xeptocore is a lightweight, host-based Intrusion Detection System (IDS) designed to identify deepfakes and AI-manipulated visual content in real-time. Operating with an out-of-band monitoring approach, it can scan active video streams from any third-party applications (such as Zoom, Google Meet, YouTube, or local media players) without requiring API integrations or webhooks.

## Core Features

*   **Triple-Model Ensemble Architecture**
    Utilizes a weighted voting mechanism combining three distinct Convolutional Neural Networks (Meso4-DF, Meso4-F2F, and MesoInception4) to achieve high accuracy and reduce false positives in real-world scenarios.
*   **Target Lock (Screen Scraping)**
    Features an intuitive "Snipping Tool" mechanism that allows users to draw a bounding box over any specific area of their monitor. The system will lock onto this physical coordinate and continuously scan the visual feed.
*   **Zero-API Dependency & Privacy-Preserving**
    Operates 100% locally on the host machine. No video data is ever transmitted to cloud servers, ensuring absolute privacy for sensitive video conferences.
*   **Lightweight CPU Execution**
    Engineered for optimal performance without the need for high-end GPUs. It runs efficiently on standard CPUs (achieving 15-20 FPS) by isolating the ROI (Region of Interest) and utilizing multithreading for the GUI and detection pipeline.
*   **Adaptive Illumination (CLAHE)**
    Integrates Contrast Limited Adaptive Histogram Equalization (CLAHE) to preprocess incoming frames, ensuring stable face detection even in poor or backlit lighting conditions.

## System Architecture

DeepGuard-IDS functions as a Blue Team preventative tool. The workflow consists of:
1.  **Frame Acquisition**: Continuous screen capture using `mss`.
2.  **Preprocessing**: CLAHE enhancement followed by Haar Cascade face isolation.
3.  **Inference**: The isolated face tensor is passed asynchronously to the Triple-Model CNN.
4.  **Scoring & Alerting**: A moving average (5-frame smoothing) calculates the final authenticity probability. If the threshold exceeds the anomaly limit, an Intrusion Alert is logged into the dashboard.

## Installation

### Prerequisites
*   Windows OS (optimized for DPI awareness)
*   Miniconda / Anaconda
*   Python 3.9

### Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/edgararya/DeepGuard-IDS.git
   cd DeepGuard-IDS
   ```

2. Create and activate a dedicated Conda environment:
   ```bash
   conda create -n deepguard python=3.9 -y
   conda activate deepguard
   ```

3. Install required dependencies:
   ```bash
   pip install opencv-python mss numpy pillow
   pip install tensorflow==2.10.0
   ```
   *Note: TensorFlow 2.10.0 is the last version to officially support native Windows GPU acceleration if you plan to configure CUDA, though this application is optimized for CPU usage.*

## Usage

1. Ensure the Conda environment is active.
2. Launch the dashboard:
   ```bash
   python live_scanner.py
   ```
3. Click the **[+] Kunci Target Layar** button.
4. Click and drag to draw a selection box over the video feed or video conference window you wish to monitor.
5. The dashboard will reappear and begin real-time analysis. The terminal panel will log any detected anomalies.

## Disclaimer

This software is developed as an academic prototype for educational and research purposes in the field of Information Security. It is intended to serve as a proof-of-concept for anomaly-based visual intrusion detection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
