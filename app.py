import os
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import cvzone

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")
names = model.names

cap = cv2.VideoCapture("rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100")

active_people = 0
entered_zone = 0
logs = []


def add_log(person_id, action):
    global logs
    logs.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "person_id": str(person_id),
            "action": action,
        }
    )
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

    cvzone.putTextRect(frame, f"Group1 Enter: {len(counted_enter)}", (50, 60), 1, 1)
    cvzone.putTextRect(frame, f"Group1 Exit: {len(counted_exit)}", (50, 80), 1, 1)
    cvzone.putTextRect(frame, f"Group2 Enter: {len(counted_enter2)}", (50, 110), 1, 1)
    cvzone.putTextRect(frame, f"Group2 Exit: {len(counted_exit2)}", (50, 130), 1, 1)

    active_people = len(counted_enter) - len(counted_exit)
    entered_zone = len(counted_enter)

    return frame


"""def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )"""


def generate():
    global cap
    rtsp_url = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100?tcp"
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, attempting to reconnect...")
            cap.release()
            cv2.waitKey(1000)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        frame = process_frame(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/statistics")
def statistics():
    global active_people, entered_zone
    return jsonify({"active_people": active_people, "entered_zone": entered_zone})


@app.route("/logs")
def get_logs():
    global logs
    return jsonify(logs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
