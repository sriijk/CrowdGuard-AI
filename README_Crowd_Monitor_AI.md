
# 🧠 Crowd Monitor AI – Real-Time Crowd Density & Risk Detection

> **Smart Surveillance for Safer Spaces**  
> Built by [Srishti Bhatnagar](https://github.com/sriijk)

An AI-powered application to detect, track, and monitor crowd density in real-time using deep learning and computer vision. Ideal for large events, public spaces, and emergency scenarios where **overcrowding** can pose serious safety threats.

---

## 🚨 Problem This Solves

Overcrowding in public areas can quickly escalate into **chaos** — causing panic, stampedes, injuries, or worse. Conventional surveillance systems are **reactive**, not **proactive**.

**Crowd Monitor AI** helps by:

- ✅ **Automatically detecting** and tracking people
- ✅ **Monitoring zone-wise density** over time
- ✅ **Raising real-time alerts** in risky conditions

This system transforms passive monitoring into **actionable intervention**, helping prevent crowd disasters before they occur.

---

## 🌟 Features

- 🧠 Real-time **YOLOv8 object detection**
- 👣 **Tracking** of individuals across frames
- 📊 **Zone-Based Risk Analysis** (Green 🟢 / Yellow 🟡 / Red 🔴)
- 🔔 **Audio Alerts** when overcrowding is detected
- 🌐 **Streamlit Web Interface** for interactive control
- 📹 Works with **video files** or **live webcam stream**

---

## 🗂️ Project Structure

```
Crowd_Monitor_AI/
├── app_utils/
│   ├── detection.py         # YOLO detection wrapper
│   ├── tracking.py          # Simple object tracking
│   ├── zone_analysis.py     # Risk zone classification
│   └── __init__.py
│
├── model/
│   └── yolov8n.pt           # Pre-trained YOLOv8n model
│
├── Beep.m4a                 # Warning sound (Yellow zone)
├── Beep2.m4a                # Alert sound (Red zone)
├── main.py                  # Runs video mode
├── test_webcam.py           # Runs webcam mode
├── streamlit_app.py         # Web-based interface (Streamlit)
├── TestVideo.mp4            # Sample test video
├── requirements.txt         # All Python dependencies
```

---

## ⚙️ Getting Started

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/sriijk/Crowd_Monitor_AI.git
cd Crowd_Monitor_AI
```

### ✅ 2. Install Required Libraries

Use pip to install dependencies:

```bash
pip install -r requirements.txt
```

### ✅ 3. Run the App

#### ▶️ Streamlit Web App:

```bash
streamlit run streamlit_app.py
```

#### ▶️ Run with Test Video:

```bash
python main.py
```

#### ▶️ Run with Webcam:

```bash
python test_webcam.py
```

---

## 🔍 How It Works

1. **YOLOv8** detects people in each frame of video/webcam
2. A **simple tracker** follows each person across frames
3. The frame is divided into **zones** (green/yellow/red)
4. Risk levels are calculated based on **people count per zone**
5. **Audio cues** alert when thresholds are crossed:
   - 🟢 Safe: No sound
   - 🟡 Warning: Beep sound
   - 🔴 Danger: Loud alert

---

## 🌐 Live Demo (Coming Soon)

> A Streamlit-hosted version will be live soon!  
> 📎 Stay tuned here: [https://github.com/sriijk/Crowd_Monitor_AI](https://github.com/sriijk/Crowd_Monitor_AI)

---

## 📦 Requirements

- Python 3.8 or higher  
- OpenCV (`cv2`)  
- Ultralytics YOLOv8  
- Streamlit  
- NumPy  
- PyTorch  

Check `requirements.txt` for the full list.

---

## 💡 Use Cases

- 🏟️ **Event Venues**: Concerts, sports, rallies  
- 🚉 **Transit Hubs**: Railway stations, airports, bus terminals  
- 🏫 **Institutions**: Schools, campuses, auditoriums  
- 🆘 **Emergency Situations**: Disaster zones, fire evacuations  
- 🛍️ **Commercial Zones**: Malls, markets, exhibitions  

---

## 🤖 Model Info

- Model: `YOLOv8n` (nano) from Ultralytics  
- Pre-trained on COCO dataset  
- Lightweight for real-time inference  
- Stored at: `model/yolov8n.pt`  

You can swap with `yolov8m.pt`, `yolov8l.pt`, or `custom.pt` based on accuracy/speed needs.

---

## 🙌 Contributing

Your ideas, bug reports, and contributions are welcome!  
To contribute:

1. Fork the repository  
2. Create a new branch  
3. Commit your changes  
4. Open a Pull Request

```bash
# Example
git fork https://github.com/sriijk/Crowd_Monitor_AI.git
git checkout -b feature-name
# make your changes
git commit -m "Add feature"
git push
```

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🧠 Author

**Srishti Bhatnagar**  
GitHub: [@sriijk](https://github.com/sriijk)

---

## 🚀 Let’s Make Public Spaces Safer with AI

> "Crowd Monitor AI isn't just a computer vision project — it's a step toward **automated crowd safety** powered by real-time analytics."  
> Stay tuned for deployment updates, screenshots, and video previews!
