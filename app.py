from flask import Flask, request, render_template, Response
import os
import cv2
import numpy as np
import subprocess
import threading

app = Flask(__name__)

font = cv2.FONT_HERSHEY_SIMPLEX
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # set video width
camera.set(4, 480)  # set video height

def run_detect():
    subprocess.run(['python', 'yolov5/detect.py', '--source', '0'], shell=True)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Warning: Failed to capture frame.")
            continue  # Skip to next frame
        else:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Warning: Failed to encode frame.")
                continue  # Skip to next frame
            frame = buffer.tobytes()
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/streaming', methods=['POST'])
def streaming():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    if request.form.get('start') == '1':
        # Run detection in a separate thread
        threading.Thread(target=run_detect).start()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
