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

# ✅ Play beep sound via HTML (Cloud-friendly)
def play_beep():
    with open("Beep2.m4a", "rb") as f:
        beep_base64 = base64.b64encode(f.read()).decode("utf-8")
    components.html(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{beep_base64}" type="audio/mp3">
    </audio>
    """, height=0)

# ✅ Updated speak() function to auto-play beep on cloud
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
        play_beep()
        print(f"TTS skipped on cloud; beep played instead: {text}")

st.set_page_config(page_title="CrowdGuardAI", layout="wide")
st.title("\U0001F6E1️ CrowdGuardAI - Real-Time Crowd Monitoring")

# Init session state
for key, val in {
    "source_mode": None, "last_beep": 0, "LOG": [],
    "peak_count": 0, "start_time": time.time(), "zone_beep_timers": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Sidebar controls
st.sidebar.header("⚙️ Control Settings")
alert_threshold = st.sidebar.slider("Overcrowding Alert Threshold (per zone)", 1, 50, 5)
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
show_boxes = st.sidebar.checkbox("Show Detection Boxes", value=True)

if st.sidebar.button("▶️ Start Webcam"):
    st.session_state.source_mode = "webcam"
if st.sidebar.button("⏹️ Stop Webcam"):
    st.session_state.source_mode = None
if st.sidebar.button("📁 Upload Video"):
    st.session_state.source_mode = "video"
if st.sidebar.button("💾 Export Logs"):
    st.session_state.source_mode = "export"
if st.sidebar.button("🔙 Back to Home"):
    st.session_state.source_mode = None

FRAME_WINDOW = st.empty()

# ✅ FINAL CODE (Only Display Metrics Reversed)
def render_stats(current_total):
    elapsed = time.time() - st.session_state.start_time
    avg_density = np.mean([
    sum([v for k, v in row.items() if k.startswith("Zone_")]) for row in reversed(st.session_state.LOG[-30:])
]) if st.session_state.LOG else 0  # Avg over last 30 logs only

    if current_total > st.session_state.peak_count:
        st.session_state.peak_count = current_total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001F465 Current", current_total)
    col2.metric("📈 Peak", st.session_state.peak_count)
    col3.metric("⏱️ Uptime", f"{int(elapsed)}s")
    col4.metric("📊 Avg Density", f"{avg_density:.1f}")


# ------------------ MODES ------------------
if st.session_state.source_mode is None:
    st.markdown("""
    This tool uses YOLOv8 for:
    - ✅ Real-time people detection from webcam or video
    - 🧶 Dynamic crowd density analysis using a 3x3 zone grid
    - 🚨 Intelligent alerts with customizable thresholds and audio warnings
    - 📈 Visual heatmaps (live, cumulative, average) and trend graphs
    - 💾 Exportable CSV logs for analysis and record keeping

    👉 Choose an option from the sidebar to start!
    """)
    st.warning("⚠️ Browser webcam access may not work in Safari. Please use Chrome or Firefox.")

elif st.session_state.source_mode == "webcam":
    st.info("⚠️ Note: If you're running this app on platforms like Hugging Face or Streamlit Cloud, browser webcam and AI speech functionality may not work. Please run this app locally to access those features.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.zone_beep_timers = {}

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            frame_h, frame_w = img.shape[:2]
            zone_counts = [0] * (GRID_ROWS * GRID_COLS)
            detections = detect_people(model, img, conf=detection_confidence)
            draw_zone_grid(img, GRID_ROWS, GRID_COLS)

            # ✅ ADD STATUS MESSAGE HERE (Webcam)
            alert_messages = []
            for i, count in enumerate(zone_counts):
                if count >= alert_threshold:
                    alert_messages.append(f"🚨 Overcrowded: People: {count}, Zone: {i}")
                elif count >= int(alert_threshold * 0.6):
                    alert_messages.append(f"⚠️ Moderate crowd: People: {count}, Zone: {i}")
            if alert_messages:
                st.warning("\n".join(alert_messages))
            else:
                st.success("🟢 Normal: All zones under control")

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

    st.success("✅ Accessing browser webcam...")
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
            st.success("✅ Webcam connected! Monitoring started...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("📴 Webcam disconnected.")
                    break

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                zone_counts = [0] * (GRID_ROWS * GRID_COLS)
                detections = detect_people(model, frame, conf=detection_confidence)
                draw_zone_grid(frame, GRID_ROWS, GRID_COLS)

                # ✅ ADD STATUS MESSAGE HERE (Video)
                alert_messages = []
                for i, count in enumerate(zone_counts):
                    if count >= alert_threshold:
                        alert_messages.append(f"🚨 Overcrowded: People: {count}, Zone: {i}")
                    elif count >= int(alert_threshold * 0.6):
                        alert_messages.append(f"⚠️ Moderate crowd: People: {count}, Zone: {i}")
                if alert_messages:
                    st.warning("\n".join(alert_messages))
                else:
                    st.success("🟢 Normal: All zones under control")

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

        st.success("🎥 Video loaded.")
        process_video(cap)

elif st.session_state.source_mode == "export":
    st.markdown("## 📊 Crowd Log Viewer")

    if st.session_state.LOG:
        df = pd.DataFrame(st.session_state.LOG)

        with st.expander("📄 Raw Log Table"):
            st.dataframe(df, use_container_width=True, height=300)

        with st.expander("📈 Crowd Trend Graph"):
            try:
                df_numeric = df.copy()
                df_numeric["timestamp"] = pd.to_datetime(df_numeric["timestamp"])
                st.line_chart(df_numeric.set_index("timestamp"), height=300)
            except Exception as e:
                st.warning(f"Trend graph couldn't be rendered: {e}")

        with st.expander("🔥 Heatmap of Last Frame"):
            try:
                last_counts = pd.to_numeric(df.iloc[-1].drop("timestamp"), errors='coerce').astype(float).values.reshape(GRID_ROWS, GRID_COLS)
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                sns.heatmap(last_counts, annot=True, cmap="YlOrRd", cbar=True, ax=ax1)
                st.pyplot(fig1)
            except Exception as e:
                st.warning(f"Heatmap render failed: {e}")

        with st.expander("🔁 Cumulative Heatmap (Sum Over Time"):
            try:
                df_numeric = df.drop("timestamp", axis=1).apply(pd.to_numeric, errors="coerce")
                sum_counts = df_numeric.sum().values.reshape(GRID_ROWS, GRID_COLS)
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                sns.heatmap(sum_counts, annot=True, cmap="YlOrRd", cbar=True, ax=ax2)
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Cumulative heatmap failed: {e}")

        with st.expander("📊 Average Heatmap (Mean Over Time"):
            try:
                avg_counts = df_numeric.mean().values.reshape(GRID_ROWS, GRID_COLS)
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                sns.heatmap(avg_counts, annot=True, cmap="YlGnBu", cbar=True, ax=ax3)
                st.pyplot(fig3)
            except Exception as e:
                st.warning(f"Average heatmap failed: {e}")

        st.download_button(
            "⬇️ Download Log as CSV",
            df.to_csv(index=False),
            file_name="crowd_log.csv",
            mime="text/csv"
        )

        st.success(f"✅ Export ready with {len(df)} records")
    else:
        st.info("No log data available yet. Try starting webcam or uploading a video.")
