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
import platform  # ✅ NEW: For OS detection

from utils import detect_people, get_zone_id, draw_zone_grid

GRID_ROWS, GRID_COLS = 3, 3
model = YOLO("yolov8n.pt")

# ✅ Updated speak() function to skip TTS on non-Windows (like Streamlit Cloud)
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
st.title("🛡️ CrowdGuardAI - Real-Time Crowd Monitoring")

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

# Display metrics
def render_stats(current_total):
    elapsed = time.time() - st.session_state.start_time
    avg_density = np.mean([
        sum([v for k, v in row.items() if k.startswith("Zone_")]) for row in st.session_state.LOG
    ]) if st.session_state.LOG else 0

    if current_total > st.session_state.peak_count:
        st.session_state.peak_count = current_total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Current", current_total)
    col2.metric("📈 Peak", st.session_state.peak_count)
    col3.metric("⏱️ Uptime", f"{int(elapsed)}s")
    col4.metric("📊 Avg Density", f"{avg_density:.1f}")

# Core processing loop

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

        for det in detections:
            x1, y1, x2, y2, conf = det
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
            zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
            zone_counts[zone_id] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (xc, yc), 3, (0, 0, 255), -1)

        total = sum(zone_counts)
        render_stats(total)

        for i, count in enumerate(zone_counts):
            if count >= alert_threshold:
                cv2.putText(frame, f"Zone {i} OVERCROWDED!", (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                last_time = st.session_state.zone_beep_timers.get(i, 0)
                if time.time() - last_time > 3:
                    if os.path.exists("Beep2.m4a"):
                        with open("Beep2.m4a", "rb") as f:
                            audio_base64 = base64.b64encode(f.read()).decode()
                            components.html(f"""
                            <audio autoplay>
                            <source src=\"data:audio/mp3;base64,{audio_base64}\" type=\"audio/mp3\">
                            </audio>
                            """, height=0)
                    speak(f"Alert! Overcrowding in zone {i}")
                    st.session_state.zone_beep_timers[i] = time.time()

        if total >= alert_threshold:
            st.error(f"🚨 Total crowd too high: {total}")
        else:
            st.success("🟢 Normal density")

        log_row = {"timestamp": time.strftime("%H:%M:%S")}
        for i, c in enumerate(zone_counts):
            log_row[f"Zone_{i}"] = c
        st.session_state.LOG.append(log_row)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, width=640, channels="RGB")  # Reduced webcam size
        time.sleep(0.03)

    cap.release()

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

elif st.session_state.source_mode == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        st.session_state.source_mode = None
    else:
        process_video(cap)

elif st.session_state.source_mode == "video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp.name)
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
    else:
        st.info("No log data available yet. Try starting webcam or uploading a video.")