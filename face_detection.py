import os
import cv2
import streamlit as st
import numpy as np

# Print a greeting message
st.write("Hello! This is a Face Detection app using the Viola-Jones algorithm.")

# Load the face cascade classifier
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# Function to detect faces
def detect_faces(frame, rectangle_color, save_images, min_neighbors, scale_factor):
    # Convert the frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Convert the color from hexadecimal to BGR format
    bgr_color = tuple(int(rectangle_color[i:i + 2], 16) for i in (1, 3, 5))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

    # Show the image with detected faces using Streamlit
    st.image(frame, channels="BGR")

    # Save the image with detected faces if enabled
    if save_images:
        # Get the project directory path
        project_path = os.path.dirname(os.path.abspath(__file__))
        # Create the 'images' folder if it doesn't exist
        images_folder = os.path.join(project_path, 'images')
        os.makedirs(images_folder, exist_ok=True)
        # Save the image with the appropriate path
        save_path = os.path.join(images_folder, "detected_faces.jpg")
        cv2.imwrite(save_path, frame)
        st.write(f"Image saved: {save_path}")


# Main app function
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the 'Detect Faces' button to start detecting faces from a test image.")
    st.write("Adjust the parameters and color below to customize the detection.")

    # Generate unique keys for the color picker widgets
    rectangle_color = st.color_picker("Choose the color for the rectangles", "#00FF00", key='color_picker1')

    # Add options for saving images
    save_images = st.checkbox("Save Images")

    # Add options to adjust minNeighbors and scaleFactor
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # Add a file uploader for the user to provide an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        if uploaded_file is not None:
            # Convert the uploaded file to a numpy array and decode it using OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            # Call the detect_faces function with the chosen parameters
            detect_faces(frame, rectangle_color, save_images, min_neighbors, scale_factor)
        else:
            st.error("Please upload an image to proceed with face detection.")


# Run the app
app()

