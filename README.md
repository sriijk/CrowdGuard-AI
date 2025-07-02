# 🛡️ CrowdGuard AI

**An AI-Powered Real-Time Crowd Monitoring and Overcrowding Alert System Using YOLOv8 and Heatmap-Based Zone Analysis**

---

## 📌 Project Summary

CrowdGuard AI is a real-time crowd monitoring and alert system built using deep learning and visual zone analysis. It leverages the lightweight **YOLOv8 Nano** model to detect people and applies a **3x3 zone grid overlay** to monitor density levels. When a zone surpasses a predefined crowd threshold, the system raises visual and audio-based alerts. The application also provides trend analytics and heatmaps to support crowd safety and planning decisions.

This tool is especially valuable for event organizers, emergency responders, school administrators, and city planners.

---

## 🌟 Key Features

* ✅ Real-time webcam or video analysis
* ✅ YOLOv8-powered people detection
* ✅ Grid-based zone tracking (3x3 matrix)
* ✅ Visual alerts for overcrowded zones
* ✅ Sound alerts for high-risk zones *(local systems only)*
* ✅ Live metrics: current, peak, average density
* ✅ Time-based line graph of density trends
* ✅ Heatmap visualizations (last frame, cumulative, average)
* ✅ Log export to CSV

---

## 💡 Voice Alert Logic

* The application attempts to speak zone-wise crowd warnings using text-to-speech (TTS) via `pyttsx3`.
* ⚠️ **Speech/voice alerts only function on local Windows systems.**
* Streamlit Cloud and other cloud platforms will skip this step with a console print instead.

---

## 📊 Functional Overview

* 🎥 Detects people from **webcam** or **uploaded video**
* 🧠 Applies YOLOv8 for accurate, real-time person detection
* 🏋️ Divides frame into 9 zones (3x3 grid) and counts people per zone
* 💥 Alerts on screen and with sound when crowd exceeds set threshold
* 📊 Generates heatmaps and trend charts to analyze density over time
* 📆 Logs detection data per zone with timestamps in CSV

---

## 🔠 Architecture

```
CrowdGuard-AI/
├── streamlit_app.py        # Streamlit interface with video analysis & UI
├── app.py                   # Local OpenCV-only version for testing
├── utils.py                 # Helper functions: YOLO detection, grid logic
├── requirements.txt         # Python dependencies
├── Beep2.m4a                # Audio alert sound file
└── yolov8n.pt               # Pre-trained YOLOv8 Nano model
```

---

## 🔹 How to Run

### Option 1: 🚀 Deploy on Streamlit Cloud (Recommended)

1. Push this project to a GitHub repository
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Set the main file to `streamlit_app2.py`
5. Click "Deploy"

> Note: Audio alerts (TTS) will not work on cloud.

### Option 2: 🔧 Run Locally (Full Features)

```bash
git clone https://github.com/yourusername/CrowdGuard-AI.git
cd CrowdGuard-AI
pip install -r requirements.txt
streamlit run streamlit_app2.py
```

> ✉️ Make sure `yolov8n.pt` and `Beep2.m4a` are in the same directory.

---

## 📊 Example Alerts

**Zone-wise Detection:**

```
Zone 0 : 1   Zone 1 : 5   Zone 2 : 2
Zone 3 : 0   Zone 4 : 7   Zone 5 : 1
Zone 6 : 1   Zone 7 : 3   Zone 8 : 2
```

> Alerts: ⚠️ Zone 4 OVERCROWDED!

**Graph & Heatmaps:**

* Real-time graph of total detections per timestamp
* Heatmaps showing last frame, cumulative crowd over time, and average density

---

## 📅 Use Cases

* 🍎 School/college campus crowd management
* 🌺 Religious event monitoring
* 🎟† Stadiums, malls, metro stations
* 🚓 Emergency planning in public gatherings
* 🗳️ Real-time surveillance dashboards

---

## 📂 Requirements

```
streamlit>=1.32
opencv-python
numpy
pandas
matplotlib
seaborn
ultralytics
torch
pyttsx3
```

---

## 👨‍💼 Developed By

**Srishti Bhatnagar**
🎓 B.Tech CSE | AI-ML & CV Developer
📧 [srishtibhatnagar051@gmail.com](mailto:srishtibhatnagar051@gmail.com)
🔗 GitHub: [@sriijk](https://github.com/sriijk)

---

## 🙏 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)

---

## ⭐ Like this Project?

Give it a ⭐ on GitHub and share it with your peers!
