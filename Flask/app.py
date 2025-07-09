from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from collections import deque, defaultdict, Counter
from datetime import datetime
import time

app = Flask(__name__)

model = YOLO("yolov8s.pt")  # Load YOLOv8 model

CAMERA_STREAMS = {
    'cam 1': "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_1/tracks-v1/index.fmp4.m3u8",
    'cam 2': "https://cctv.balitower.co.id/Gelora-017-700470_2/index.fmp4.m3u8",
    'cam 3': "https://cctv.balitower.co.id/Gelora-017-700470_3/tracks-v1/index.fmp4.m3u8",
    'cam 4': "https://cctv.balitower.co.id/Tomang-004-702108_2/index.fmp4.m3u8",
    'cam 5': "https://cctv.balitower.co.id/Jati-Pulo-001-702017_2/tracks-v1/index.fmp4.m3u8",
    'cam 6': "https://cctv.balitower.co.id/Cikoko-006-705651_4/index.fmp4.m3u8"
}

current_camera = {'id': 'cam 1'}

# store vehicle counts per minute
vehicle_stats = defaultdict(lambda: defaultdict(list))  # vehicle_stats[camera_id][minute] = list of counts

@app.route('/vehicle_data')
def vehicle_data():
    cam_id = current_camera['id']
    stats = vehicle_stats[cam_id]
    aggregated = []
    for minute, counts in sorted(stats.items())[-30:]:  # last 30 minutes
        avg_count = sum(counts) / len(counts) if counts else 0
        aggregated.append({
            "time": minute,
            "count": round(avg_count, 2)
        })
    return jsonify(aggregated)


@app.route('/')
def index():
    return render_template('index.html', current=current_camera['id'], year=datetime.now().year)

@app.route('/set_camera/<cam_id>')
def set_camera(cam_id):
    if cam_id in CAMERA_STREAMS:
        current_camera['id'] = cam_id
    return render_template('index.html', current=current_camera['id'], year=datetime.now().year)

@app.route('/video_feed')
def video_feed():
    stream_url = CAMERA_STREAMS[current_camera['id']]
    return Response(generate_frames(stream_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

from collections import deque
import time


def generate_frames(url):
    status_buffer = deque(maxlen=7)
    frame_buffer = deque(maxlen=5)  # Keep only last 5 frames
    
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

        # Add frame to buffer
        frame_buffer.append(frame.copy())
        
        # Always process and serve frames if buffer has frames
        if frame_buffer:
            # Use the oldest frame in buffer for processing
            processing_frame = frame_buffer.popleft()
            
            results = model(processing_frame, verbose=False)
            annotated = processing_frame.copy()

            vehicle_count = 0
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = names[cls_id]
                if class_name not in vehicle_classes:
                    continue

                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"
                
                # Define custom colors for each class
                box_colors = {
                    'car': (0, 255, 0),         # Green
                    'motorcycle': (255, 0, 0),  # Blue
                    'bus': (0, 165, 255),       # Orange
                    'truck': (128, 0, 128)      # Purple
                }

                color = box_colors.get(class_name, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                text_bg_tl = (x1, y1 - text_height - 10)
                text_bg_br = (x1 + text_width + 6, y1)
                cv2.rectangle(annotated, text_bg_tl, text_bg_br, color, thickness=-1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Traffic status
            if vehicle_count == 0:
                current_status = "  No vehicles"
            elif vehicle_count <= 10:
                current_status = "  Less traffic"
            else:
                current_status = "  Crowded"

            status_buffer.append(current_status)
            most_common_status = Counter(status_buffer).most_common(1)[0][0] if status_buffer else current_status
            color = (0, 0, 255) if most_common_status == "Crowded" else (0, 200, 0) if most_common_status == "Less traffic" else (200, 200, 200)

            minute_key = datetime.now().strftime("%H:%M")
            vehicle_stats[current_camera['id']][minute_key].append(vehicle_count)

            cv2.rectangle(annotated, (10, 10), (350, 70), (0, 0, 0), -1)
            cv2.putText(annotated, f"Vehicles: {int(vehicle_count)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, most_common_status, (160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            ret, buffer = cv2.imencode('.jpg', annotated)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Small delay to prevent overwhelming the CPU
        time.sleep(0.033)  # ~30 FPS

if __name__ == '__main__':
    print("🚀 Running YOLOv8 Flask App")
    app.run(debug=True, threaded=True)