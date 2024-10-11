import os
import cv2
import streamlit as st

# Print a greeting message
st.write("Hello! This is a Face Detection app using the Viola-Jones algorithm.")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Global variable to control the detection loop
is_running = False

# Function to detect faces
def detect_faces(rectangle_color, save_images, min_neighbors, scale_factor):
    global is_running
    is_running = True
    counter = 0

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while is_running:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Convert the color from hexadecimal to BGR format
        bgr_color = tuple(int(rectangle_color[i:i + 2], 16) for i in (1, 3, 5))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Check if faces are detected
        if len(faces) > 0:
            # Show the image with detected faces using Streamlit
            st.image(frame, channels="BGR")

            # Save the image with detected faces if enabled
            if save_images:
                counter += 1
                # Get the project directory path
                project_path = os.path.dirname(os.path.abspath(__file__))
                # Create the 'images' folder if it doesn't exist
                images_folder = os.path.join(project_path, 'images')
                os.makedirs(images_folder, exist_ok=True)
                # Save the image with the appropriate path
                save_path = os.path.join(images_folder, f"image_{counter}.jpg")
                cv2.imwrite(save_path, frame)
                st.write(f"Image saved: {save_path}")

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    # Add instructions to the Streamlit app interface
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the 'Detect Faces' button to start detecting faces from your webcam.")
    st.write("To stop the detection, press 'q' on your keyboard.")
    st.write("Adjust the parameters and color below to customize the detection.")

    # Generate unique keys for the color picker widgets
    rectangle_color = st.color_picker("Choose the color for the rectangles", "#00FF00", key='color_picker1')

    # Add options for saving images
    save_images = st.checkbox("Save Images")

    # Add options to adjust minNeighbors and scaleFactor
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function with the chosen parameters
        detect_faces(rectangle_color, save_images, min_neighbors, scale_factor)

    # Add a button to stop the face detection
    if st.button("Stop Detection"):
        global is_running
        is_running = False

app()

