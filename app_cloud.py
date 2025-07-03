import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import streamlit.components.v1 as components
import os
import base64
import pyttsx3
import platform
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from utils import detect_people, get_zone_id, draw_zone_grid

GRID_ROWS, GRID_COLS = 3, 3
model = YOLO("yolov8n.pt")

def speak(text):
    if platform.system() == "Windows":
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    else:
        print(f"TTS skipped on non-Windows system: {text}")

st.set_page_config(page_title="CrowdGuardAI", layout="wide")
st.title("\U0001F6E1️ CrowdGuardAI - Real-Time Crowd Monitoring")

for key, val in {
    "source_mode": None, "last_beep": 0, "LOG": [],
    "peak_count": 0, "start_time": time.time(), "zone_beep_timers": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.sidebar.header("\u2699\ufe0f Control Settings")
alert_threshold = st.sidebar.slider("Overcrowding Alert Threshold (per zone)", 1, 50, 5)
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
show_boxes = st.sidebar.checkbox("Show Detection Boxes", value=True)

if st.sidebar.button("\u25b6\ufe0f Start Webcam"):
    st.session_state.source_mode = "webcam"
if st.sidebar.button("\u23f9\ufe0f Stop Webcam"):
    st.session_state.source_mode = None
if st.sidebar.button("\ud83d\udcc1 Upload Video"):
    st.session_state.source_mode = "video"
if st.sidebar.button("\ud83d\udcbe Export Logs"):
    st.session_state.source_mode = "export"
if st.sidebar.button("\ud83d\udd19 Back to Home"):
    st.session_state.source_mode = None

FRAME_WINDOW = st.empty()

def render_stats(current_total):
    elapsed = time.time() - st.session_state.start_time
    avg_density = np.mean([
        sum([v for k, v in row.items() if k.startswith("Zone_")]) for row in st.session_state.LOG
    ]) if st.session_state.LOG else 0

    if current_total > st.session_state.peak_count:
        st.session_state.peak_count = current_total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001F465 Current", current_total)
    col2.metric("\ud83d\udcc8 Peak", st.session_state.peak_count)
    col3.metric("\u23f1\ufe0f Uptime", f"{int(elapsed)}s")
    col4.metric("\ud83d\udcca Avg Density", f"{avg_density:.1f}")

if st.session_state.source_mode is None:
    st.markdown("""
    This tool uses YOLOv8 for:
    - \u2705 Real-time people detection from webcam or video
    - \ud83e\uddf6 Dynamic crowd density analysis using a 3x3 zone grid
    - \ud83d\udea8 Intelligent alerts with customizable thresholds and audio warnings
    - \ud83d\udcc8 Visual heatmaps (live, cumulative, average) and trend graphs
    - \ud83d\udcbe Exportable CSV logs for analysis and record keeping

    \ud83d\udc49 Choose an option from the sidebar to start!
    """)
    st.warning("\u26a0\ufe0f Browser webcam access may not work in Safari. Please use Chrome or Firefox.")

elif st.session_state.source_mode == "webcam":
    st.info("\u26a0\ufe0f Note: If you're running this app on platforms like Hugging Face or Streamlit Cloud, browser webcam and AI speech functionality may not work. Please run this app locally to access those features.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.zone_beep_timers = {}

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            frame_h, frame_w = img.shape[:2]
            zone_counts = [0] * (GRID_ROWS * GRID_COLS)
            detections = detect_people(model, img, conf=detection_confidence)
            draw_zone_grid(img, GRID_ROWS, GRID_COLS)

            for det in detections:
                x1, y1, x2, y2, conf = det
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
                zone_counts[zone_id] += 1
                if show_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (xc, yc), 3, (0, 0, 255), -1)

            total = sum(zone_counts)
            render_stats(total)

            alert_messages = []
            for i, count in enumerate(zone_counts):
                if count >= alert_threshold:
                    alert_messages.append(f"\ud83d\udea8 Overcrowded: People: {count}, Zone: {i}")
                elif count >= int(alert_threshold * 0.6):
                    alert_messages.append(f"\u26a0\ufe0f Moderate crowd: People: {count}, Zone: {i}")

            if alert_messages:
                st.warning("\n".join(alert_messages))
            else:
                st.success("\ud83d\udfe2 Normal: All zones under control")

            for i, count in enumerate(zone_counts):
                if count >= alert_threshold:
                    cv2.putText(img, f"Zone {i} OVERCROWDED!", (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if time.time() - self.zone_beep_timers.get(i, 0) > 3:
                        speak(f"Alert! Overcrowding in zone {i}")
                        self.zone_beep_timers[i] = time.time()

            log_row = {"timestamp": time.strftime("%H:%M:%S")}
            for i, c in enumerate(zone_counts):
                log_row[f"Zone_{i}"] = c
            st.session_state.LOG.append(log_row)

            if len(st.session_state.LOG) > 500:
                st.session_state.LOG = st.session_state.LOG[-500:]

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.success("\u2705 Accessing browser webcam...")
    webrtc_streamer(
        key="live",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif st.session_state.source_mode == "video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp.name)

        def process_video(cap):
            st.success("\u2705 Webcam connected! Monitoring started...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("\ud83d\udcf4 Webcam disconnected.")
                    break

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                zone_counts = [0] * (GRID_ROWS * GRID_COLS)
                detections = detect_people(model, frame, conf=detection_confidence)
                draw_zone_grid(frame, GRID_ROWS, GRID_COLS)

                for det in detections:
                    x1, y1, x2, y2, conf = det
                    xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                    zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
                    zone_counts[zone_id] += 1
                    if show_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (xc, yc), 3, (0, 0, 255), -1)

                total = sum(zone_counts)
                render_stats(total)

                alert_messages = []
                for i, count in enumerate(zone_counts):
                    if count >= alert_threshold:
                        alert_messages.append(f"\ud83d\udea8 Overcrowded: People: {count}, Zone: {i}")
                    elif count >= int(alert_threshold * 0.6):
                        alert_messages.append(f"\u26a0\ufe0f Moderate crowd: People: {count}, Zone: {i}")

                if alert_messages:
                    st.warning("\n".join(alert_messages))
                else:
                    st.success("\ud83d\udfe2 Normal: All zones under control")

                for i, count in enumerate(zone_counts):
                    if count >= alert_threshold:
                        cv2.putText(frame, f"Zone {i} OVERCROWDED!", (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if time.time() - st.session_state.zone_beep_timers.get(i, 0) > 3:
                            speak(f"Alert! Overcrowding in zone {i}")
                            st.session_state.zone_beep_timers[i] = time.time()

                log_row = {"timestamp": time.strftime("%H:%M:%S")}
                for i, c in enumerate(zone_counts):
                    log_row[f"Zone_{i}"] = c
                st.session_state.LOG.append(log_row)

                if len(st.session_state.LOG) > 500:
                    st.session_state.LOG = st.session_state.LOG[-500:]

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame, width=640, channels="RGB")
                time.sleep(0.03)

            cap.release()

        st.success("\ud83c\udfa5 Video loaded.")
        process_video(cap)

elif st.session_state.source_mode == "export":
    st.markdown("## \ud83d\udcca Crowd Log Viewer")

    if st.session_state.LOG:
        df = pd.DataFrame(st.session_state.LOG[::-1])

        with st.expander("\ud83d\udcc4 Raw Log Table"):
            st.dataframe(df, use_container_width=True, height=300)

        with st.expander("\ud83d\udcc8 Crowd Trend Graph"):
            try:
                df_numeric = df.copy()
                df_numeric["timestamp"] = pd.to_datetime(df_numeric["timestamp"])
                st.line_chart(df_numeric.set_index("timestamp"), height=300)
            except Exception as e:
                st.warning(f"Trend graph couldn't be rendered: {e}")

        st.download_button(
            "\u2b07\ufe0f Download Log as CSV",
            df.to_csv(index=False),
            file_name="crowd_log.csv",
            mime="text/csv"
        )

        st.success(f"\u2705 Export ready with {len(df)} records")
    else:
        st.info("No log data available yet. Try starting webcam or uploading a video.")
