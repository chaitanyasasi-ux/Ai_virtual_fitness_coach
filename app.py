import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os
import socket
from workouts.bicep_curl import bicep_curl_tracking, both_arm_tracking
from utils.pose_utils import calculate_angle

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Fitness Coach - Bicep Curl", layout="wide")
st.title("💪 AI Virtual Fitness Coach - Bicep Curl")

# ---------- Detect if Running on Streamlit Cloud ----------
def running_in_cloud():
    hostname = socket.gethostname()
    return "streamlit" in hostname.lower() or os.environ.get("HOME") == "/home/adminuser"

# ---------- Pose Utils ----------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- Session State ----------
if 'counter' not in st.session_state:
    st.session_state.counter = 0
    st.session_state.stage = None
    st.session_state.feedback = ""
    st.session_state.logs = []
    st.session_state.sets_completed = 0
    st.session_state.rep_start_time = None
    st.session_state.rep_times = []
    st.session_state.sets_stats = []

# ---------- Sidebar Controls ----------
st.sidebar.markdown("## ⏳ Workout Settings")
duration = st.sidebar.slider("Workout Duration for a set (minutes)", 1, 10, 2)
sets = st.sidebar.slider("Number of Sets", 1, 5, 3)
min_rep_time = st.sidebar.slider("Min Rep Time (s) - Too Fast", 0.0, 1.0, 0.4, 0.1)
mode = st.sidebar.radio("Select Mode:", ["Right Arm", "Left Arm", "Both Arms"])
start_button = st.sidebar.button("🚀 Start Workout")

# ---------- Video Source ----------
def get_video_capture():
    if running_in_cloud():
        st.warning("⚠️ Streamlit Cloud does not support webcam. Upload a video instead.")
        uploaded_file = st.file_uploader("Upload your workout video", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            return cv2.VideoCapture(tfile.name)
        else:
            return None
    else:
        return cv2.VideoCapture(0)

# ---------- Layout Setup ----------
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
rep_placeholder = col2.empty()
avg_placeholder = col2.empty()
feedback_placeholder = col2.empty()
countdown_placeholder = col2.empty()
set_placeholder = col2.empty()

# ---------- Rep Tracking Logic ----------
def track_rep(angle, low_thresh, high_thresh):
    if angle > high_thresh:
        st.session_state.stage = "down"
        st.session_state.rep_start_time = time.time()
    if angle < low_thresh and st.session_state.stage == "down":
        rep_time = time.time() - st.session_state.rep_start_time if st.session_state.rep_start_time else 0
        st.session_state.stage = "up"
        st.session_state.counter += 1
        st.session_state.rep_times.append(rep_time)
        st.session_state.logs.append((datetime.now(), mode, st.session_state.counter, round(rep_time, 2)))

        if rep_time < min_rep_time:
            st.session_state.feedback = "⚠️ Too fast — slow down."
            return False
        else:
            st.session_state.feedback = "✅ Nice rep!"
            return True
    return None

# ---------- Start Workout Logic ----------
if start_button:
    st.session_state.sets_stats = []
    cap = get_video_capture()

    if cap is None or not cap.isOpened():
        st.error("❌ Could not access video source.")
    else:
        for set_num in range(sets):
            st.session_state.feedback = ""
            st.session_state.counter = 0
            st.session_state.rep_times = []
            good_reps = 0
            bad_reps = 0
            end_time = time.time() + duration * 60

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened() and time.time() < end_time:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if not running_in_cloud():
                        frame = cv2.flip(frame, 1)

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        if results.pose_landmarks is None:
                            st.session_state.feedback = "⚠️ Pose not detected. Adjust your position."
                            continue

                        landmarks = results.pose_landmarks.landmark
                        height, width, _ = image.shape

                        if mode == "Right Arm" or mode == "Left Arm":
                            angle = bicep_curl_tracking(landmarks, mode)
                        else:
                            angle = both_arm_tracking(landmarks)

                        result = track_rep(angle, 60, 160)
                        if result is True:
                            good_reps += 1
                        elif result is False:
                            bad_reps += 1

                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    except Exception as e:
                        st.session_state.feedback = "⚠️ Error occurred: pose not detected."
                        print("Error:", e)

                    avg_time = round(np.mean(st.session_state.rep_times), 2) if st.session_state.rep_times else 0
                    remaining = int(end_time - time.time())
                    mins, secs = divmod(remaining, 60)

                    countdown_placeholder.markdown(f"### ⏱️ Time Left: `{mins:02d}:{secs:02d}`")
                    rep_placeholder.markdown(f"### 🔢 Reps: `{st.session_state.counter}`")
                    avg_placeholder.markdown(f"### 📊 Avg Rep Time: `{avg_time}s`")
                    feedback_placeholder.markdown(f"### 💬 Feedback: {st.session_state.feedback}")
                    set_placeholder.markdown(f"### 🧹 Set: `{set_num + 1}/{sets}`")
                    frame_placeholder.image(image, channels="BGR")

            st.session_state.sets_stats.append({"good": good_reps, "bad": bad_reps})

            fig, ax = plt.subplots()
            ax.bar(["Good Reps", "Bad Reps"], [good_reps, bad_reps], color=["green", "red"])
            ax.set_title(f"Set {set_num + 1} Rep Quality")
            ax.set_ylim(0, max(good_reps + bad_reps, 5))
            st.pyplot(fig)

            if set_num < sets - 1:
                st.info(f"⏸️ Resting 3 minutes before next set...")
                for rest in range(3 * 60, 0, -1):
                    mins, secs = divmod(rest, 60)
                    countdown_placeholder.markdown(f"### 💥 Rest Time: `{mins:02d}:{secs:02d}`")
                    time.sleep(1)

        st.success("✅ Workout Complete!")
        df = pd.DataFrame(st.session_state.logs, columns=["Timestamp", "Mode", "Reps", "Rep Duration (s)"])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📅 Download Workout Log", csv, "workout_log.csv", "text/csv")

        cap.release()
