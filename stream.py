import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np

animal_model = YOLO("animal_model/best.pt")
human_model = YOLO("person_model/best.pt")

st.set_page_config(page_title="Real-Time Detection", layout="wide")

if not st.session_state.get('popup_shown', False):
    st.session_state['popup_shown'] = True
    with st.container():
        st.info(
            """
            ### 📢 Welcome to the Real-Time Human & Animal Detection Dashboard! 🦁🚶
            
            This application helps you monitor live video feeds to detect and count humans and animals in real time. It’s designed for use cases like farm protection, wildlife monitoring, and security surveillance.
            
            #### Key Features:
            - 🧠 **Dual Detection Modes:** Run human and animal detection simultaneously or individually.
            - 📷 **Dynamic Camera Selection:** Automatically detects available cameras, with options to switch between built-in and external cameras.
            - 📲 **Mobile Camera Support:** Stream video directly from your phone using a simple URL.
            - 🔢 **Real-Time Count Updates:** View the current number of humans and animals detected, updated live.
            - ✅ **Confidence-Based Detection:** Animal detections are only shown when confidence is above 60% for higher accuracy.
            - ⏯ **Start/Stop Controls:** Easily start and stop detection for each model independently.
            
            Enjoy exploring your live streams with precise and responsive detection! 
            """
        )
        if st.button("Close Overview"):
            st.session_state['popup_shown'] = False
            st.experimental_rerun()

st.title("📸 Real-Time Detection Dashboard")

def get_available_cameras():
    available_cameras = {}
    built_in_found = False
    external_count = 1

    for index in range(0, 5): 
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                if not built_in_found:
                    available_cameras[f"External Camera {external_count}"] = index
                    external_count += 1
                else:
                    available_cameras["Built-in Webcam"] = index
                    built_in_found = True
    return available_cameras

camera_options = get_available_cameras()

col1, col2 = st.columns(2)

with col1:
    st.header("🦁 Animal Detection")
    video_source_animal = st.radio("Select Video Source (Animal):", ("Webcam", "Mobile Camera"), key="animal")

    if video_source_animal == "Webcam":
     selected_camera_animal = st.selectbox("Select Camera:", list(camera_options.keys()), index=0)
    
    if selected_camera_animal in camera_options:
        video_source_animal = camera_options[selected_camera_animal]
    else:
        st.warning("Selected camera not found. Using default camera.")
        video_source_animal = 0

    start_animal = st.button("Start Animal Detection")
    stop_animal = st.button("Stop Animal Detection")

    if start_animal:
        cap = cv2.VideoCapture(video_source_animal)
        stframe_animal = st.empty()
        animal_count_placeholder = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success or stop_animal:
                break

            animal_results = animal_model(frame)[0]
            animal_count = sum(1 for box in animal_results.boxes if box.conf[0].item() >= 0.6)

            for box in animal_results.boxes:
                confidence = box.conf[0].item()
                if confidence >= 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = animal_results.names[int(box.cls)]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe_animal.image(frame, channels="RGB", use_column_width=True)
            animal_count_placeholder.write(f"Current number of animals detected: {animal_count}")

        cap.release()

with col2:
    st.header("🚶 Human Detection")
    video_source_human = st.radio("Select Video Source (Human):", ("Webcam", "Mobile Camera"), key="human")

    if video_source_human == "Webcam":
     selected_camera_human = st.selectbox("Select Camera:", list(camera_options.keys()), index=0, key="human_cam")
    
    # Handle missing camera safely
    if selected_camera_human in camera_options:
        video_source_human = camera_options[selected_camera_human]
    else:
        st.warning("Selected camera not found. Using default camera.")
        video_source_human = 0  # Default to built-in camera


    start_human = st.button("Start Human Detection")
    stop_human = st.button("Stop Human Detection")

    if start_human:
        cap = cv2.VideoCapture(video_source_human)
        stframe_human = st.empty()
        human_count_placeholder = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success or stop_human:
                break

            human_results = human_model(frame)[0]
            human_count = sum(1 for box in human_results.boxes if human_results.names[int(box.cls)] == "person")

            for box in human_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = human_results.names[int(box.cls)]
                confidence = box.conf[0].item()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe_human.image(frame, channels="RGB", use_column_width=True)
            human_count_placeholder.write(f"Current number of humans detected: {human_count}")

        cap.release()

st.success("Detection app ready! Choose a video source and start detecting.")
