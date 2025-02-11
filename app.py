import os
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import cvzone
import pyrebase
import subprocess
import time
import threading

from dotenv import load_dotenv

load_dotenv()

firebaseConfig = {
    "apiKey": "AIzaSyBvR_ldDK5y1HTi1UrKUbzHcQWNDknM09o",
    "authDomain": "mycash-ai.firebaseapp.com",
    "databaseURL": "https://mycash-ai-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "mycash-ai",
    "storageBucket": "mycash-ai.firebasestorage.app",
    "messagingSenderId": "757415222278",
    "appId": "1:757415222278:web:2f73933891270feba33787",
    "measurementId": "G-8LFV5DN71J",
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

app = Flask(__name__)
CORS(app)
app.config["APPLICATION_ROOT"] = "/office-ai"

model = YOLO("best.pt")
names = model.names

rtsp_url = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100?tcp"

active_people = 0
entered_zone = 0
logs = []


def add_log(person_id, action):
    global logs
    log_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": str(person_id),
        "action": action,
    }
    logs.append(log_data)
    db.child("logs").push(log_data)
    if len(logs) > 100:
        logs.pop(0)


enter = {}
exit = {}
counted_enter = []
counted_exit = []

enter2 = {}
exit2 = {}
counted_enter2 = []
counted_exit2 = []

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    "1020x600",
    "-r",
    "15",
    "-i",
    "-",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-b:v",
    "1000k",
    "-f",
    "flv",
    "rtmp://localhost/live/office",
    "-f",
    "hls",
    "-hls_time",
    "2",
    "-hls_list_size",
    "10",
    "-hls_flags",
    "delete_segments",
    "/var/www/html/hls/office.m3u8",
]



def process_frame(frame):
    global enter, exit, counted_enter, counted_exit
    global enter2, exit2, counted_enter2, counted_exit2
    global active_people, entered_zone
    frame = cv2.resize(frame, (1020, 600))
    area1 = [(327, 292), (322, 328), (730, 328), (730, 292)]
    area2 = [(322, 336), (312, 372), (730, 372), (730, 336)]
    area3 = [(338, 78), (338, 98), (720, 98), (720, 78)]
    area4 = [(341, 107), (341, 130), (720, 130), (720, 107)]
    results = model.track(frame, persist=True)
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        for box, class_id, track_id, conf in zip(
            boxes, class_ids, track_ids, confidences
        ):
            c = names[class_id]
            x1, y1, x2, y2 = box
            point = (x1, y2)
            if "Person" in c:
                result0 = cv2.pointPolygonTest(np.array(area1, np.int32), point, False)
                if result0 >= 0:
                    enter[track_id] = point
                if track_id in enter:
                    result1 = cv2.pointPolygonTest(
                        np.array(area2, np.int32), point, False
                    )
                    if result1 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, f"{track_id}", (x1, y1), 1, 1)
                        cv2.circle(frame, point, 4, (255, 0, 0), -1)
                        if track_id not in counted_enter:
                            counted_enter.append(track_id)
                result02 = cv2.pointPolygonTest(np.array(area2, np.int32), point, False)
                if result02 >= 0:
                    exit[track_id] = point
                if track_id in exit:
                    result03 = cv2.pointPolygonTest(
                        np.array(area1, np.int32), point, False
                    )
                    if result03 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, f"{track_id}", (x1, y1), 1, 1)
                        cv2.circle(frame, point, 4, (255, 0, 0), -1)
                        if track_id not in counted_exit:
                            counted_exit.append(track_id)
            if "P1" in c or "P2" in c:
                result2 = cv2.pointPolygonTest(np.array(area3, np.int32), point, False)
                if result2 >= 0:
                    enter2[track_id] = point
                if track_id in enter2:
                    result3 = cv2.pointPolygonTest(
                        np.array(area4, np.int32), point, False
                    )
                    if result3 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, f"{track_id}", (x1, y1), 1, 1)
                        cv2.circle(frame, point, 4, (255, 0, 0), -1)
                        if track_id not in counted_enter2:
                            counted_enter2.append(track_id)
                            add_log(c, "Entered Zone")
                result22 = cv2.pointPolygonTest(np.array(area4, np.int32), point, False)
                if result22 >= 0:
                    exit2[track_id] = point
                if track_id in exit2:
                    result33 = cv2.pointPolygonTest(
                        np.array(area3, np.int32), point, False
                    )
                    if result33 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, f"{track_id}", (x1, y1), 1, 1)
                        cv2.circle(frame, point, 4, (255, 0, 0), -1)
                        if track_id not in counted_exit2:
                            counted_exit2.append(track_id)
                            add_log(c, "Exited Zone")

    active_people = len(counted_enter) - len(counted_exit)
    entered_zone = len(counted_enter)
    return frame


def generate():
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("ERROR: Unable to open RTSP stream. Check camera URL.")
        return
    frame_skip = 3
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, attempting to reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        frame = process_frame(frame)
        if frame is None:
            continue
        ffmpeg_process.stdin.write(frame.tobytes())


#@app.route("/video_feed")
#def video_feed():
#    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/statistics")
def statistics():
    global active_people, entered_zone
    stats = {"active_people": active_people, "entered_zone": entered_zone}
    db.child("statistics").set(stats)
    return jsonify(stats)


@app.route("/logs")
def get_logs():
    global logs
    return jsonify(logs)


if __name__ == "__main__":
    capture_thread = threading.Thread(target=generate, daemon=True)
    capture_thread.start()
    app.run(host="0.0.0.0", port=5001)

