
# 🛡️ CrowdGuard AI  
**An AI-Powered Real-Time Crowd Monitoring and Overcrowding Alert System Using YOLOv8 and Heatmap-Based Zone Analysis**

---

## 📌 Project Summary

CrowdGuard AI is a real-time crowd monitoring and safety system designed to detect overcrowded zones in public areas using a combination of deep learning and zone-based spatial analysis. It uses **YOLOv8** for object detection and applies **heatmap-based zone logic** to trigger alerts when crowd density exceeds safe thresholds.

This system is especially beneficial for event organizers, public safety officials, and emergency response teams to preemptively manage dangerous crowd situations.

---

## 🎯 Objectives

- Prevent crowd disasters by early detection of overcrowded zones
- Provide zone-wise visualization for better crowd management
- Allow deployment in real-time settings like schools, religious events, festivals, and stadiums
- Offer alerting mechanisms (sound, visual) for high-risk detection
- Provide trend tracking via graphs and heatmap evolution

---

## 🔍 What It Does

- 🎥 Accepts **webcam input** or **uploaded video**
- 🧠 Detects people using **YOLOv8 Nano model**
- 🧱 Applies **3x3 zone grid overlay** on each frame
- 🔥 Calculates live and cumulative heatmaps for density monitoring
- 📈 Plots real-time graphs showing crowd evolution over time
- 🚨 Triggers **audio alerts** and visual warnings when zones are overcrowded
- 📄 Logs timestamped alerts and zone values in CSV for analysis

---

## 🖼️ Key Features

- ✅ YOLOv8-powered real-time people detection
- ✅ Grid-based zone analysis (3x3 matrix)
- ✅ Heatmap rendering: last frame, cumulative, average
- ✅ Real-time overcrowding alert (sound + color zone highlight)
- ✅ Upload `.mp4` or use webcam directly
- ✅ Log file export of alert history
- ✅ Crowd density line chart with timestamp
- ✅ Customizable light/dark UI theme with responsive layout

---

## 🌐 Live Deployment

> 🚀 **Hosted on Replit**: [Launch CrowdGuard AI →](https://replit.com/@your-username/CrowdGuardAI)  
> *(Replace with your actual live link)*

---

## 📦 Folder Structure

```
CrowdGuard-AI/
├── app_utils/
│   ├── detection.py          # YOLOv8 detection wrapper
│   └── zone_analysis.py      # Zone heatmap and density calculator
├── Beep.m4a / Beep2.m4a      # Audio alert sound
├── app.py                    # Main Logic
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # All required libraries
├── model/
|   └──yolov8n.pt             # YOLOv8 Nano model 
├── TestVideo.mp4             # Sample video for testing
└── README.md                 # This file
```

---

## 📚 YOLOv8 Model Info

- Model used: [`yolov8n.pt`](https://github.com/ultralytics/ultralytics)
- Download link:
  ```
  wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
  ```
- Trained on COCO dataset
- Lightweight (ideal for real-time edge deployment)

> You may substitute with `yolov8s.pt`, `yolov8m.pt`, or custom-trained model based on deployment needs.

---

## 🧪 Sample Output

**Detected Zones with Alert:**
```
[ 2  1  4 ]
[ 0  8  2 ]
[ 1  3  1 ]
```
> This means Zone 5 (center) has a high density (alert triggered).

---

## 📊 Real-Time Graphs

- 📈 X-axis: Frame number or timestamp  
- 📉 Y-axis: Total people detected or per-zone count  
- Line chart automatically updates with each frame

---

## 🔊 Sound Alert Logic

- Beep triggers if any zone exceeds a configurable threshold (default: 5)
- Zone with highest crowd density is highlighted in red
- Audio alert plays from the browser (Replit-supported)

---

## 💡 Real-World Use Cases

- 🕌 Religious festivals with massive crowd gathering
- 🏟️ Sports stadiums & concerts
- 🛍️ Malls, metro stations, and transit hubs
- 🏫 School/university events
- 🚨 Protest sites or public demonstrations

---

## 🛠️ How to Use

### ▶️ Online via Replit
1. Go to: [https://replit.com/@your-username/CrowdGuardAI](https://replit.com/@your-username/CrowdGuardAI)
2. Click “Run”
3. Choose webcam mode or upload video
4. Watch real-time detection, graphs, and alerts

### ▶️ Locally
```bash
git clone https://github.com/yourusername/CrowdGuard-AI.git
cd CrowdGuard-AI
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📋 Requirements

```
ultralytics
opencv-python
streamlit
pandas
matplotlib
numpy
seaborn
```

---

## 🌐 Why Replit for Deployment?

Unlike most cloud platforms (Render, Streamlit Cloud, Hugging Face), Replit:

- ✅ Supports webcam access inside browser  
- ✅ Supports sound alerts through media files  
- ✅ Runs both backend (OpenCV + YOLO) and frontend in one space  

This makes Replit ideal for demoing all features of **CrowdGuard AI** without stripping any functionality.

---

## 👨‍💻 Developed By

**Srishti Bhatnagar**  
🎓 B.Tech in CSE  
💡 AI & Computer Vision Enthusiast  
📬 srishtibhatnagar051@gmail.com  
🔗 GitHub: [@sriijk](https://github.com/sriijk)

---

## 🙏 Acknowledgements

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)  
- [Streamlit UI Framework](https://streamlit.io/)  
- [OpenCV](https://opencv.org/)  

---

## ⭐ Star the Repo
> If you liked this project, consider giving it a ⭐ on GitHub to support the developer.
