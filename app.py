import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from workouts.bicep_curl import bicep_curl_tracking, both_arm_tracking
from utils.pose_utils import calculate_angle

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Fitness Coach - Bicep Curl", layout="wide")
st.title("💪 AI Virtual Fitness Coach - Bicep Curl")

mode = st.radio("Select Mode:", ["Right Arm", "Left Arm", "Both Arms"])

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
    st.session_state.sets_stats = []  # to store good/bad reps per set

# ---------- Sidebar Controls ----------
st.sidebar.markdown("## ⏳ Workout Settings")
duration = st.sidebar.slider("Workout Duration for a set (minutes)", 1, 10, 2)
sets = st.sidebar.slider("Number of Sets", 1, 5, 3)
min_rep_time = st.sidebar.slider("Min Rep Time (s) - Too Fast", 0.0, 0.5, 1.0, 0.1)
start_button = st.sidebar.button("🚀 Start Workout")

# ---------- Layout Setup ----------
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
rep_placeholder = col2.empty()
avg_placeholder = col2.empty()
feedback_placeholder = col2.empty()
countdown_placeholder = col2.empty()
set_placeholder = col2.empty()

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
    return None  # no rep counted

if start_button:
    st.session_state.sets_stats = []  # reset stats before workout

    for set_num in range(sets):
        st.session_state.feedback = ""
        st.session_state.counter = 0
        st.session_state.rep_times = []
        good_reps = 0
        bad_reps = 0
        end_time = time.time() + duration * 60
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    break

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
                    elif mode == "Both Arms":
                        angle = both_arm_tracking(landmarks)
                    else:
                        angle = 0  # fallback

                    result = track_rep(angle, 60, 160)
                    if result is True:
                        good_reps += 1
                    elif result is False:
                        bad_reps += 1

                    # Draw angles
                    if mode == "Right Arm" or mode == "Left Arm":
                        if mode == "Right Arm":
                            shoulder = [int(landmarks[12].x * width), int(landmarks[12].y * height)]
                            elbow = [int(landmarks[14].x * width), int(landmarks[14].y * height)]
                            wrist = [int(landmarks[16].x * width), int(landmarks[16].y * height)]
                        else:
                            shoulder = [int(landmarks[11].x * width), int(landmarks[11].y * height)]
                            elbow = [int(landmarks[13].x * width), int(landmarks[13].y * height)]
                            wrist = [int(landmarks[15].x * width), int(landmarks[15].y * height)]

                        elbow_angle = int(np.round(angle))
                        cv2.putText(image, f'Elbow: {elbow_angle}°', tuple(elbow),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

                    elif mode == "Both Arms":
                        # Left arm
                        left_shoulder = [int(landmarks[12].x * width), int(landmarks[12].y * height)]
                        left_elbow = [int(landmarks[14].x * width), int(landmarks[14].y * height)]
                        left_wrist = [int(landmarks[16].x * width), int(landmarks[16].y * height)]
                        left_elbow_angle = int(np.round(calculate_angle(left_shoulder, left_elbow, left_wrist)))
                        cv2.putText(image, f'L Elbow: {left_elbow_angle}°', tuple(left_elbow),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                        # Right arm
                        right_shoulder = [int(landmarks[11].x * width), int(landmarks[11].y * height)]
                        right_elbow = [int(landmarks[13].x * width), int(landmarks[13].y * height)]
                        right_wrist = [int(landmarks[15].x * width), int(landmarks[15].y * height)]
                        right_elbow_angle = int(np.round(calculate_angle(right_shoulder, right_elbow, right_wrist)))
                        cv2.putText(image, f'R Elbow: {right_elbow_angle}°', tuple(right_elbow),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

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

            cap.release()

        # After set ends, store stats
        st.session_state.sets_stats.append({"good": good_reps, "bad": bad_reps})

        # Plot reps quality for this set
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
