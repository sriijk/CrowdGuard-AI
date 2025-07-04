# 🛡️ CrowdGuard AI  
**An AI-Powered Real-Time Crowd Monitoring and Overcrowding Alert System Using YOLOv8 and Heatmap-Based Zone Analysis**

---

## 📌 Project Summary
CrowdGuard AI is an AI-powered real-time crowd monitoring and alert system designed to enhance safety, surveillance, and crowd management using computer vision.

✅ Key highlights:
- ⚡ Real-time Crowd Detection: Leverages YOLOv8 Nano for fast, efficient people detection on live video streams or uploaded videos.
- 🗺️ Zone-Based Density Analysis: Splits video frames into a 3×3 grid to monitor people density within each zone.
- 🚨 Instant Alerts: Automatically generates visual and audio alerts if crowd levels exceed defined safety thresholds.
- 📊 Analytics & Heatmaps: Offers crowd trend graphs and heatmaps (live, cumulative, and average) for deep analysis.
- 💾 Data Logging & Export: Saves crowd data with timestamps, exportable as CSV for further reporting.

✅ Suitable for use in:
- 🎪 Large Events & Public Gatherings: Festivals, concerts, rallies, exhibitions.
- 🏫 School & Campus Safety: Monitor canteens, hallways, or assembly areas.
- 🚆 Transportation Hubs: Metro stations, airports, bus terminals.
- 🏬 Commercial Spaces: Shopping malls, markets, or stadiums.
- 🚨 Emergency Evacuation Planning: Monitor crowd buildup during drills or emergencies.

---

## 🌟 Key Features  
- ✅ Real-time webcam or video input  
- ✅ YOLOv8-powered people detection  
- ✅ Grid-based zone tracking (3x3 matrix)  
- ✅ Visual alerts for overcrowded zones  
- ✅ Sound alerts (on supported systems)  
- ✅ Zone-specific crowd alert logs (Latest logs show on top)  
- ✅ Real-time metrics: Current, Peak, Average Density, Uptime  
- ✅ Trend Graphs (crowd over time)  
- ✅ Heatmaps: Last Frame, Cumulative, and Average  
- ✅ CSV Export of Detection Logs  

---

## 🔔 Audio & Webcam Compatibility Notes
- 🔊 The app uses Text-to-Speech (TTS) powered by pyttsx3 for zone-specific crowd alerts. This feature works only on local systems (such as Windows).
- 💻 On cloud platforms like Streamlit Cloud or Hugging Face Spaces, webcam access and audio alerts are not supported due to browser and platform restrictions. However, a video upload option is provided as an alternative, and a webcam option is also available—if it works in your browser, you can use it.
- ✅ To experience the app with full functionality (including real-time webcam access and voice alerts), it is recommended to run the app locally.

---

## 📊 Functional Overview
- 🎥 Detects people from webcam or uploaded video.
- 🧠 Uses YOLOv8 for real-time person detection.
- 🏙️ Divides the frame into 9 zones (3x3 grid) and counts people per zone.
- 🚨 Auto-alerts (visual + audio) when crowd exceeds the defined threshold.
- 📈 Trend graphs and heatmaps help analyze crowd patterns over time.
- 📝 Logs per-zone counts with timestamps into CSV for analysis.

---

## 🔠 Project Structure

```
CrowdGuard-AI/
├── app_local.py          # Local system version (full features: webcam, video upload, TTS audio & beep alerts)
├── app_cloud.py          # Streamlit Cloud & Hugging Face Spaces version (video upload + beep alerts only)
├── app.py                # Legacy OpenCV-only version for testing (CLI-based)
├── utils.py              # Helper functions (YOLO detection, grid logic, etc.)
├── requirements.txt      # Python dependencies
├── Beep2.m4a             # Audio alert sound file (beep)
└── yolov8n.pt            # Pre-trained YOLOv8 Nano model

```


---

## 📚 YOLOv8 Model Info
- **Model:** `yolov8n.pt`  
- **Download Link:** [Download YOLOv8 Nano Model](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- **Dataset:** COCO  
- **Size:** ~6 MB  
- **Purpose:** Ultra-fast, lightweight real-time person detection.

⚠️ **Note:** The model file is not included in this repo. Please **download manually** and place it inside the root folder.

---

## 🔹 How to Run  
### 🚀 Try Live Demos (Video Upload Only, View-Only)
> ⚠️ Webcam & audio alerts won’t work on cloud platforms.

- 🌐 **[Streamlit Cloud Demo → Try Now](https://crowdguard-ai-xxjwxh56aazz35csz975yy.streamlit.app/)**
- 🤗 **[Hugging Face Spaces Demo → Try Now](https://huggingface.co/spaces/SrishtiB/CrowdGuardAI)**

---

### 🔧 Local Run (Full Features Recommended)
```bash
git clone https://github.com/yourusername/CrowdGuard-AI.git
cd CrowdGuard-AI
pip install -r requirements.txt
streamlit run streamlit_app.py

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
🎓 B.Tech CSE | AI & Computer Vision Enthusiast | Machine Learning & Deep Learning

📧 [srishtibhatnagar051@gmail.com](mailto:srishtibhatnagar051@gmail.com)

🔗 GitHub: [@sriijk](https://github.com/sriijk)

🔗 LinkedIn : [Srishti Bhatnagar](www.linkedin.com/in/srishti-bhatnagar-b59833269)

---

## 🙏 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)

---

## ⭐ Like this Project?

If you find this project useful, please give it a ⭐ on GitHub and share it with others!
