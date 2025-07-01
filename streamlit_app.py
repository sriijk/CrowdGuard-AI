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

from app_utils.detection import detect_people
from app_utils.zone_analysis import get_zone_id, draw_zone_grid

# ------------------ CONFIG ------------------
GRID_ROWS, GRID_COLS = 3, 3
LOG = []
model = YOLO("yolov8n.pt")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Crowd Monitor AI", layout="wide")
st.title("🎥 AI-Powered Crowd Density & Risk Monitor")

# ------------------ SESSION STATE INIT ------------------
if "source_mode" not in st.session_state:
    st.session_state.source_mode = None
if "last_beep" not in st.session_state:
    st.session_state.last_beep = 0
# ✅ Add this line to fix the error
if "LOG" not in st.session_state:
    st.session_state.LOG = []

# ------------------ SIDEBAR ------------------
st.sidebar.header("🎛️ Controls")
threshold = st.sidebar.slider("⚠️ Zone Threshold", 1, 10, 4)

if st.sidebar.button("▶️ Start Webcam"):
    st.session_state.source_mode = "webcam"

if st.sidebar.button("⏹️ Stop Webcam"):
    st.session_state.source_mode = None

if st.sidebar.button("📁 Upload Video"):
    st.session_state.source_mode = "video"

if st.sidebar.button("💾 Export Logs"):
    st.session_state.source_mode = "export"

# ------------------ UI ------------------
FRAME_WINDOW = st.empty()

# ---------- No Source Selected ----------
if st.session_state.source_mode is None:
    st.markdown("""
    ### 👋 Welcome to Crowd Monitor AI

    This tool uses YOLOv8 to:
    - Detect people in real-time
    - Analyze crowd density in a 3x3 grid
    - Trigger alerts with sound if overcrowding is detected
    - Visualize trends with graphs and heatmaps
    - Export logs to CSV

    👉 Choose an option from the sidebar to start!
    """)

# ---------- Webcam Mode ----------
elif st.session_state.source_mode == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        st.session_state.source_mode = None
    else:
        st.success("✅ Webcam connected.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Couldn't read from webcam.")
                continue  # don't break; wait for valid frame

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            zone_counts = [0] * (GRID_ROWS * GRID_COLS)

            detections = detect_people(model, frame)
            draw_zone_grid(frame, GRID_ROWS, GRID_COLS)

            for det in detections:
                x1, y1, x2, y2, conf = det
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
                zone_counts[zone_id] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (xc, yc), 3, (0, 0, 255), -1)

            # Show counts
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    zone_id = i * GRID_COLS + j
                    x = j * frame_w // GRID_COLS + 5
                    y = i * frame_h // GRID_ROWS + 20
                    color = (0, 0, 255) if zone_counts[zone_id] >= threshold else (255, 255, 255)
                    cv2.putText(frame, f'{zone_counts[zone_id]}', (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Overcrowding alert
            alert_zones = [i for i, count in enumerate(zone_counts) if count >= threshold]
            print("Zone Counts:", zone_counts)
            print("Alert Zones:", alert_zones)

            if alert_zones:
                st.error(f"🚨 Overcrowding in zones: {', '.join(map(str, alert_zones))}")

                # Play alert sound (only once every 2 sec)
                if time.time() - st.session_state.last_beep > 2:
                    audio_file = "Beep2.m4a"
                    if os.path.exists(audio_file):
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                            audio_base64 = base64.b64encode(audio_bytes).decode()
                            components.html(f"""
                                <audio autoplay>
                                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                                </audio>
                            """, height=0)

                            st.session_state.last_beep = time.time()

            # Logging
            log_row = {"timestamp": time.strftime("%H:%M:%S")}
            for i, count in enumerate(zone_counts):
                log_row[f"Zone_{i}"] = count
            st.session_state.LOG.append(log_row)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

# ---------- Video Upload Mode ----------
elif st.session_state.source_mode == "video":
    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st.success("🎞️ Video loaded.")

        FRAME_WINDOW = st.empty()  # Define video display container

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            zone_counts = [0] * (GRID_ROWS * GRID_COLS)

            detections = detect_people(model, frame)
            draw_zone_grid(frame, GRID_ROWS, GRID_COLS)

            for det in detections:
                x1, y1, x2, y2, conf = det
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
                zone_counts[zone_id] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (xc, yc), 3, (0, 0, 255), -1)

            # Show zone count
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    zone_id = i * GRID_COLS + j
                    x = j * frame_w // GRID_COLS + 5
                    y = i * frame_h // GRID_ROWS + 20
                    color = (0, 0, 255) if zone_counts[zone_id] >= threshold else (255, 255, 255)
                    cv2.putText(frame, f"{zone_counts[zone_id]}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Overcrowding alert
            alert_zones = [i for i, count in enumerate(zone_counts) if count >= threshold]
            print("Zone Counts:", zone_counts)
            print("Alert Zones:", alert_zones)

            if alert_zones:
                st.error(f"🚨 Overcrowding in zones: {', '.join(map(str, alert_zones))}")

                # Play alert sound (only once every 2 sec)
                if time.time() - st.session_state.last_beep > 2:
                    audio_file = "Beep2.m4a"
                    if os.path.exists(audio_file):
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                            audio_base64 = base64.b64encode(audio_bytes).decode()
                            components.html(f"""
                                <audio autoplay>
                                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                                </audio>
                            """, height=0)
                        st.session_state.last_beep = time.time()

            # Logging
            log_row = {"timestamp": time.strftime("%H:%M:%S")}
            for i, count in enumerate(zone_counts):
                log_row[f"Zone_{i}"] = count
                st.session_state.LOG.append(log_row)

            # Convert to RGB and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame, channels="RGB", use_container_width=True)

            time.sleep(0.03)  # Optional: Slow down to simulate normal video speed

        cap.release()


elif st.session_state.source_mode == "export":
    st.markdown("## 📊 Crowd Log Viewer")

    if st.session_state.LOG:
        df = pd.DataFrame(st.session_state.LOG)

        with st.expander("📄 Raw Log Table"):
            st.dataframe(df, use_container_width=True)

        with st.expander("📈 Crowd Trend Graph"):
            try:
                df_numeric = df.copy()
                df_numeric["timestamp"] = pd.to_datetime(df_numeric["timestamp"])
                st.line_chart(df_numeric.set_index("timestamp"))
            except Exception as e:
                st.warning(f"Trend graph couldn't be rendered: {e}")

        with st.expander("🔥 Heatmap of Last Frame"):
            try:
                last_counts = pd.to_numeric(df.iloc[-1].drop("timestamp"), errors='coerce').astype(float).values.reshape(GRID_ROWS, GRID_COLS)
                fig1, ax1 = plt.subplots()
                sns.heatmap(last_counts, annot=True, cmap="YlOrRd", cbar=True, ax=ax1)
                st.pyplot(fig1)
            except Exception as e:
                st.warning(f"Heatmap render failed: {e}")

        with st.expander("🔁 Cumulative Heatmap (Sum Over Time)"):
            try:
                df_numeric = df.drop("timestamp", axis=1).apply(pd.to_numeric, errors="coerce")
                sum_counts = df_numeric.sum().values.reshape(GRID_ROWS, GRID_COLS)
                fig2, ax2 = plt.subplots()
                sns.heatmap(sum_counts, annot=True, cmap="YlOrRd", cbar=True, ax=ax2)
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Cumulative heatmap failed: {e}")

        with st.expander("📊 Average Heatmap (Mean Over Time)"):
            try:
                avg_counts = df_numeric.mean().values.reshape(GRID_ROWS, GRID_COLS)
                fig3, ax3 = plt.subplots()
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
