import streamlit as st
import cv2
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
# try:
#     from mediapipe.python.solutions import face_mesh as mp_face_mesh
#     from mediapipe.python.solutions import hands as mp_hands
# except ImportError:
#     import mediapipe as mp

#     mp_face_mesh = mp.solutions.face_mesh
#     mp_hands = mp.solutions.hands
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


from utils.storage import add_session, get_sessions

st.set_page_config(page_title="FirstSigns", layout="centered")
st.title("FirstSigns - Multi-Session Screening")

child_id = st.text_input("Enter Child ID")
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

if uploaded_file and child_id:

    st.video(uploaded_file)

    if st.button("Analyze & Save Session"):

        with st.spinner("Processing..."):

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp.name)

            if not cap.isOpened():
                st.error("Video file could not be read. Please try a different file.")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            FRAME_SKIP = 2

            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)
            hands = mp_hands.Hands(static_image_mode=False)

            total_frames = 0
            face_detected = 0

            movement_series = []
            engagement_series = []
            gaze_scores = []
            gesture_series = []

            prev_center = None
            tracked_center = None

            prev_hand_center = None
            hand_persistence = 0

            frame_center = np.array([width/2, height/2])
            MAX_TRACK_DISTANCE = 120

            frame_count = 0

            while True:

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                total_frames += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_results = face_mesh.process(rgb)
                hand_results = hands.process(rgb)

                face_found = False
                gaze_frame_score = 0
                gesture_frame = 0

                if hand_results.multi_hand_landmarks:

                    xs = []
                    ys = []

                    for lm in hand_results.multi_hand_landmarks[0].landmark:
                        xs.append(lm.x * width)
                        ys.append(lm.y * height)

                    hand_center = np.array([np.mean(xs), np.mean(ys)])

                    if prev_hand_center is not None:

                        movement = np.linalg.norm(hand_center - prev_hand_center)

                        if movement > 10:
                            hand_persistence += 1
                        else:
                            hand_persistence = 0

                        if hand_persistence >= 3:
                            gesture_frame = 1

                    prev_hand_center = hand_center

                gesture_series.append(gesture_frame)

                if face_results.multi_face_landmarks:

                    chosen_center = None
                    chosen_face = None
                    min_dist = float("inf")

                    for face_landmarks in face_results.multi_face_landmarks:

                        xs = [lm.x * width for lm in face_landmarks.landmark]
                        ys = [lm.y * height for lm in face_landmarks.landmark]
                        center = np.array([np.mean(xs), np.mean(ys)])

                        dist = np.linalg.norm(center - (tracked_center if tracked_center is not None else frame_center))

                        if dist < min_dist:
                            min_dist = dist
                            chosen_center = center
                            chosen_face = face_landmarks

                    if tracked_center is not None and min_dist > MAX_TRACK_DISTANCE:
                        engagement_series.append(0)
                        continue

                    if chosen_face:

                        face_found = True
                        face_detected += 1
                        engagement_series.append(1)

                        if prev_center is not None:
                            movement_series.append(np.linalg.norm(chosen_center - prev_center))

                        prev_center = chosen_center
                        tracked_center = chosen_center

                        left_eye = chosen_face.landmark[33]
                        right_eye = chosen_face.landmark[263]
                        nose = chosen_face.landmark[1]

                        eye_mid = (left_eye.x + right_eye.x) / 2

                        head_offset = abs(nose.x - eye_mid)
                        head_align = max(0, 1 - head_offset * 10)

                        top = chosen_face.landmark[159]
                        bottom = chosen_face.landmark[145]

                        eye_open = abs(top.y - bottom.y) * 15
                        eye_open = np.clip(eye_open, 0, 1)

                        gaze_frame_score = head_align * eye_open

                gaze_scores.append(gaze_frame_score)

                if not face_found:
                    engagement_series.append(0)

            cap.release()

            if not engagement_series or not gesture_series or not gaze_scores:
                st.error("Could not extract data from video. Please use a well-lit video with a clearly visible face.")
                st.stop()

            face_presence = face_detected / total_frames if total_frames else 0
            disengagement = np.mean(np.array(engagement_series) == 0)

            gesture_score = np.mean(gesture_series)
            gaze_score = np.mean(gaze_scores)

            if movement_series:

                arr = np.array(movement_series)
                arr = np.clip(arr, 0, np.percentile(arr, 95))
                arr = np.convolve(arr, np.ones(5)/5, mode="same")

                max_val = np.max(arr)
                arr = arr / max_val if max_val > 0 else arr

                threshold = np.mean(arr) + 1.5*np.std(arr)
                spike_density = np.sum(arr > threshold) / len(arr)

                engagement = (
                    0.35 * (1 - spike_density) +
                    0.25 * gesture_score +
                    0.20 * face_presence +
                    0.20 * gaze_score
                )

            else:
                spike_density = 0
                engagement = 0

            feature_vector = [
                float(engagement),
                float(gaze_score),
                float(gesture_score),
                float(spike_density),
                float(face_presence)
            ]

            add_session(child_id,{
                "engagement":float(engagement),
                "face_presence":float(face_presence),
                "spike_density":float(spike_density),
                "disengagement":float(disengagement),
                "gaze_score":float(gaze_score),
                "gesture_score":float(gesture_score),
                "features":feature_vector
            })

            sessions = get_sessions(child_id)
            scores=[s["engagement"] for s in sessions]

            # -------- Z-SCORE BEHAVIORAL DEVIATION --------

            session_count=len(sessions)
            labels=["Engagement","Gaze","Gesture","Movement","Face Presence"]
            metric_keys=["engagement","gaze_score","gesture_score","spike_density","face_presence"]
            baseline_sessions=sessions[:-1]
            current_session=sessions[-1]

            st.subheader("Behavioral Deviation Analysis")
            st.write("Z-score analysis identifies which specific metrics deviated. Isolation Forest evaluates whether the overall behavioral profile of this session is anomalous.")

            if session_count<3:
                st.info("Insufficient session history")
            else:
                baseline=np.array([
                    [float(s[key]) for key in metric_keys]
                    for s in baseline_sessions
                ])

                current=np.array([float(current_session[key]) for key in metric_keys])

                means=np.mean(baseline,axis=0)
                stds=np.std(baseline,axis=0)

                z_scores=[]

                for current_value,mean_value,std_value in zip(current,means,stds):
                    if std_value>0:
                        z_scores.append((current_value-mean_value)/std_value)
                    elif np.isclose(current_value,mean_value):
                        z_scores.append(0.0)
                    else:
                        z_scores.append(3.0 if current_value>mean_value else -3.0)

                z_scores=np.array(z_scores)
                bdi=float(np.mean(np.abs(z_scores)))
                notable_deviation=bdi>1.5 or np.any(np.abs(z_scores)>2.0)

                st.metric("BDI (Deviation)",f"{bdi:.3f}")

                if notable_deviation:
                    st.error("Notable Deviation")
                else:
                    st.success("Within expected range")

                fig,ax=plt.subplots()

                ax.bar(labels,z_scores)

                ax.axhline(2.0,linestyle="--",color="red")
                ax.axhline(-2.0,linestyle="--",color="red")

                ax.set_ylabel("Z-score")
                ax.set_title("Current Session vs Prior Baseline")

                st.pyplot(fig)

            # -------- ISOLATION FOREST ANOMALY DETECTION --------

            st.subheader("Anomaly Detection (Isolation Forest)")

            if session_count<3:
                st.info("Insufficient data for anomaly detection")
            else:
                baseline=np.array([
                    [float(s[key]) for key in metric_keys]
                    for s in baseline_sessions
                ])

                current=np.array([[float(current_session[key]) for key in metric_keys]])

                detector=IsolationForest(contamination="auto",random_state=42)
                detector.fit(baseline)

                if_prediction=detector.predict(current)[0]
                if_score=float(detector.decision_function(current)[0])

                st.metric("Behavioral Anomaly Score",f"{if_score:.3f}")

                if if_prediction==-1:
                    st.error("Session flagged as anomalous by Isolation Forest")
                else:
                    st.success("Within normal range")

            st.subheader("Session Metrics")

            col1,col2,col3=st.columns(3)
            col1.metric("Engagement",f"{engagement:.2f}")
            col2.metric("Face Presence",f"{face_presence:.2f}")
            col3.metric("Disengagement",f"{disengagement:.2f}")

            col4,col5,col6=st.columns(3)
            col4.metric("Spike Density",f"{spike_density:.2f}")
            col5.metric("Gaze Score",f"{gaze_score:.2f}")
            col6.metric("Gesture Score",f"{gesture_score:.2f}")

            if len(scores)>1:

                st.subheader("Longitudinal Trend")

                fig3,ax3=plt.subplots()
                ax3.plot(scores,marker='o')
                ax3.set_xlabel("Session")
                ax3.set_ylabel("Engagement")

                st.pyplot(fig3)
            
            # -------- SESSION HISTORY TABLE --------

            st.subheader("Session History")

            import pandas as pd

            table_data = []

            for i, s in enumerate(sessions):

                table_data.append({
                    "Session": i+1,
                    "Engagement": round(s["engagement"],3),
                    "Gaze": round(s["gaze_score"],3),
                    "Gesture": round(s["gesture_score"],3),
                    "Movement": round(s["spike_density"],3),
                    "Face Presence": round(s["face_presence"],3)
                })

            df = pd.DataFrame(table_data)

            df.index = range(1, len(df) + 1)

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.warning("Not a diagnosis.")
