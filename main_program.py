import cv2
import numpy as np
from PIL import Image

# function to convert the OpenCV image into a PIL image
def cv2_to_pil(img):
    """
    Params:
            img: OpenCV BGR image
    Returns:
            pil_img: PIL image
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil_img

# initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def take_photo(filename='photo.jpg'):
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the resulting frame
        cv2.imshow('Press "Space" to Capture', frame)

        # Wait for the user to press 'space' to capture the image
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    # Get face bounding box coordinates using Haar Cascade
    faces = face_cascade.detectMultiScale(gray)
    
    # Draw face bounding box on image
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(filename, frame)

    # Convert the OpenCV image to a PIL image and display it
    pil_img = cv2_to_pil(frame)
    pil_img.show()

    return filename

try:
    filename = take_photo('photo.jpg')
    print('Saved to {}'.format(filename))
except Exception as err:
    print(str(err))
