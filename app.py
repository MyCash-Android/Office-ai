import os
import cv2
import numpy as np
import threading
import time
import subprocess
from datetime import datetime
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cvzone
import pyrebase
import requests
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

active_people = 0
entered_zone = 0
logs = []

enter = {}
exit = {}
counted_enter = []
counted_exit = []
enter2 = {}
exit2 = {}
counted_enter2 = []
counted_exit2 = []
cards_given = 0

def process_frame(frame, frame_count, frame_skip):
    global enter, exit, counted_enter, counted_exit
    global enter2, exit2, counted_enter2, counted_exit2
    global active_people, entered_zone
    global cards_given
    frame = cv2.resize(frame, (1020, 600))
    area1 = [(327, 292), (322, 328), (880, 328), (880, 292)]
    area2 = [(322, 336), (312, 372), (880, 372), (880, 336)]
    area3 = [(338, 78), (338, 98), (870, 98), (870, 78)]
    area4 = [(341, 107), (341, 130), (870, 130), (870, 107)]
    results = model.track(frame, persist=True)
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            point = (x1, y2)
            if "Person" in c:
                result0 = cv2.pointPolygonTest(np.array(area1, np.int32), point, False)
                if result0 >= 0:
                    enter[track_id] = point
                    print("Area 1")
                if track_id in enter:
                    result1 = cv2.pointPolygonTest(np.array(area2, np.int32), point, False)
                    if result1 >= 0:
                        print("Area 2")
                        if track_id not in counted_enter:
                            counted_enter.append(track_id)
                        #counted_enter[track_id] = counted_enter.get(track_id, 0) + 1

                result02 = cv2.pointPolygonTest(np.array(area2, np.int32), point, False)
                if result02 >= 0:
                    exit[track_id] = point
                    print("Area 2")
                if track_id in exit:
                    result03 = cv2.pointPolygonTest(np.array(area1, np.int32), point, False)
                    if result03 >= 0:
                        print("Area 1")
                        if track_id not in counted_exit:
                            counted_exit.append(track_id)
                        #counted_exit[track_id] = counted_exit.get(track_id, 0) + 1
                        if track_id in enter and track_id in exit:
                            del enter[track_id]
                            del exit[track_id]

            if "P1" in c or "P2" in c:
                result2 = cv2.pointPolygonTest(np.array(area3, np.int32), point, False)
                if result2 >= 0:
                    enter2[track_id] = point
                    print("Area 3")
                if track_id in enter2:
                    result3 = cv2.pointPolygonTest(np.array(area4, np.int32), point, False)
                    if result3 >= 0:
                        print("Area 4")
                        if track_id not in counted_enter2:
                            counted_enter2.append(track_id)
                            if c == 'P1': c = 31
                            if c == 'P2': c = 32
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            api_url = "https://backai.mycashtest.com/apiAdmin/employee/create_log"
                            params = {
                                "employee_id": c,
                                "type": 1,
                                "date": current_time
                            }   
                            try:
                                response = requests.post(api_url, params=params)
                                response.raise_for_status()  
                                print(f"Log added successfully: {response.text}")
                            except requests.RequestException as e:
                                print(f"Error adding log: {e}")
                        #counted_enter2[track_id] = counted_enter2.get(track_id, 0) + 1
                        #add_log(c, 1)

                result22 = cv2.pointPolygonTest(np.array(area4, np.int32), point, False)
                if result22 >= 0:
                    exit2[track_id] = point
                    print("Area 4")
                if track_id in exit2:
                    result33 = cv2.pointPolygonTest(np.array(area3, np.int32), point, False)
                    if result33 >= 0:
                        print("Area 3")
                        if track_id not in counted_exit2:
                            counted_exit2.append(track_id)
                            if c == 'P1': c = 31
                            if c == 'P2': c = 32
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            api_url = "https://backai.mycashtest.com/apiAdmin/employee/create_log"
                            params = {
                                "employee_id": c,
                                "type": 2,
                                "date": current_time
                            }
                            try:
                                response = requests.post(api_url, params=params)
                                response.raise_for_status()  
                                print(f"Log added successfully: {response.text}")
                            except requests.RequestException as e:
                                print(f"Error adding log: {e}")
                        #counted_exit2[track_id] = counted_exit2.get(track_id, 0) + 1
                        #add_log(c, 2)
                        if track_id in enter2 and track_id in exit2:
                            del enter2[track_id]
                            del exit2[track_id]

            if "Card" in c:
                cards_given += 1
    active_people = (len(counted_enter) - len(counted_exit))
    entered_zone = len(counted_enter)

    stats = {"active_people": active_people, "entered_zone": entered_zone}
    card = {"Number of cards given": cards_given}
    try:
        print("Updating Firebase with:", stats, card)
        db.child("statistics").set(stats)
        db.child("cards").set(card)
    except Exception as e:
        print(f"Error updating Firebase: {e}")

"""latest_frame = None
frame_lock = threading.Lock()

def update_latest_frame(frame):
    global latest_frame
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return
    frame_bytes = buffer.tobytes()
    with frame_lock:
        latest_frame = frame_bytes"""

"""@app.route("/video_feed")
def video_feed():
    def generate_frames():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)  # Adjust for desired frame rate
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")"""

@app.route("/statistics")
def statistics():
    global active_people, entered_zone
    stats = {"active_people": active_people, "entered_zone": entered_zone}
    return jsonify(stats)

@app.route("/logs")
def get_logs():
    global logs
    return jsonify(logs)

@app.route("/cards")
def cards():
    global cards_given
    card = {"Number of cards given": cards_given}
    return jsonify(card)

if __name__ == "__main__":
    from streamer import generate as start_streaming
    producer_thread = threading.Thread(target=start_streaming, daemon=True)
    producer_thread.start()

    app.run(host="0.0.0.0", port=5001, threaded=True)
