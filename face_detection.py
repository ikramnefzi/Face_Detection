import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import numpy as np
import os

# Load the face cascade classifier
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Class to handle the video stream and face detection
class FaceDetection(VideoTransformerBase):
    def __init__(self, rectangle_color, min_neighbors, scale_factor):
        self.face_cascade = face_cascade
        self.rectangle_color = tuple(int(rectangle_color[i:i + 2], 16) for i in (1, 3, 5))  # Hex to BGR
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), self.rectangle_color, 2)

        return img

# Streamlit app interface
def app():
    st.title("Face Detection using Webcam and Viola-Jones Algorithm")
    st.write("Press the 'Detect Faces' button to start detecting faces from your webcam feed.")
    st.write("Adjust the parameters and color below to customize the detection.")

    # Color picker for rectangle color
    rectangle_color = st.color_picker("Choose the color for the rectangles", "#00FF00")

    # Option to adjust minNeighbors and scaleFactor
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # Add options for saving images
    save_images = st.checkbox("Save Images")

    # Initialize webcam stream and face detection
    if st.button("Detect Faces"):
        webrtc_streamer(key="face_detection",
                        video_transformer_factory=lambda: FaceDetection(rectangle_color, min_neighbors, scale_factor))

    if save_images:
        st.write("Images will be saved in the current project directory.")

# Run the app
app()

