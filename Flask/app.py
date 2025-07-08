from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from collections import deque, Counter

app = Flask(__name__)

# Load YOLOv8 model (make sure yolov8n.pt is downloaded)
model = YOLO("yolov8n.pt")  # or "cpu" if no GPU

# Stream URLs
CAMERA_STREAMS = {
    'cam 1': "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_1/tracks-v1/index.fmp4.m3u8",
    'cam 2': "https://cctv.balitower.co.id/Gelora-017-700470_2/index.fmp4.m3u8",
    'cam 3': "https://cctv.balitower.co.id/Gelora-017-700470_3/tracks-v1/index.fmp4.m3u8",
    'cam 4': "https://cctv.balitower.co.id/Tomang-004-702108_2/index.fmp4.m3u8",
    'cam 5': "https://cctv.balitower.co.id/Jati-Pulo-001-702017_2/tracks-v1/index.fmp4.m3u8"
}

# Default selected camera
current_camera = {'id': 'cam 1'}

from datetime import datetime

@app.route('/')
def index():
    return render_template('index.html', current=current_camera['id'], year=datetime.now().year)

@app.route('/set_camera/<cam_id>')
def set_camera(cam_id):
    if cam_id in CAMERA_STREAMS:
        current_camera['id'] = cam_id
    now = datetime.now().strftime("%d %b %Y")
    return render_template('index.html', current=current_camera['id'], today=now)

@app.route('/video_feed')
def video_feed():
    stream_url = CAMERA_STREAMS[current_camera['id']]
    return Response(generate_frames(stream_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(url):
    status_buffer = deque(maxlen=7)  # Rolling window of last 15 frames
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("❌ Cannot open stream:", url)
        return

    vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
    names = model.names

    while True:
        success, frame = cap.read()
        if not success:
            continue

        results = model(frame, verbose=False)
        annotated = frame.copy()

        vehicle_count = 0  # 🧮 initialize counter

        vehicle_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            if class_name not in vehicle_classes:
                continue  # skip drawing non-vehicle classes

            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # Determine this frame's traffic status (not final display yet)
        if vehicle_count == 0:
            current_status = "  No vehicles"
        elif vehicle_count <= 10:
            current_status = "  Less traffic"
        else:
            current_status = "  Crowded"

        # Append to buffer
        status_buffer.append(current_status)

        # Use majority voting from the buffer to determine final status
        if status_buffer:
            most_common_status = Counter(status_buffer).most_common(1)[0][0]
        else:
            most_common_status = current_status  # fallback if buffer is empty

        # Set label color
        if most_common_status == "Crowded":
            color = (0, 0, 255)
        elif most_common_status == "Less traffic":
            color = (0, 200, 0)
        else:
            color = (200, 200, 200)

        # 🖼 Overlay vehicle count and status
        cv2.rectangle(annotated, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.putText(annotated, f"Vehicles: {int(vehicle_count)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, most_common_status, (160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    print("🚀 Running YOLOv8 Flask App")
    app.run(debug=True)
