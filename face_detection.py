import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define the face detection transformer class
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self, rectangle_color, min_neighbors, scale_factor):
        self.rectangle_color = rectangle_color
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor

    def transform(self, frame):
        # Convert frame to an OpenCV image
        img = frame.to_ndarray(format="bgr24")

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)

        # Convert the rectangle color from hex to BGR
        bgr_color = tuple(int(self.rectangle_color[i:i + 2], 16) for i in (1, 3, 5))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), bgr_color, 2)

        return img

# Streamlit app interface
def app():
    st.title("Face Detection using Viola-Jones Algorithm")

    # Add color picker for rectangle color
    rectangle_color = st.color_picker("Choose the color for the rectangles", "#00FF00")

    # Add sliders for minNeighbors and scaleFactor
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # Start the webcam stream using streamlit-webrtc
    webrtc_streamer(key="face-detection",
                    video_transformer_factory=lambda: FaceDetectionTransformer(rectangle_color, min_neighbors, scale_factor))

app()
