import os
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from flask import Flask
from flask_cors import CORS
from ultralytics import YOLO
import pyrebase
import requests
from dotenv import load_dotenv
import queue
import logging
from threading import Lock
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Firebase configuration
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

# Model configuration
model = YOLO("best.pt", verbose=False)


# Tracking variables
TRACK_HISTORY = 100  # frames to remember track IDs
enter = deque(maxlen=TRACK_HISTORY)
exit = deque(maxlen=TRACK_HISTORY)
counted_enter = set()
counted_exit = set()
enter2 = deque(maxlen=TRACK_HISTORY)
exit2 = deque(maxlen=TRACK_HISTORY)
counted_enter2 = set()
counted_exit2 = set()
cards_given = 0
frame_queue = queue.Queue(maxsize=10)
stats_lock = Lock()
class PeopleTracker:
    def __init__(self):
        self.names = model.names
        self.active_people = 0
        self.entered_zone = 0
        self.logs = []

        # Tracking variables using deque for better performance
        self.enter = deque(maxlen=100)
        self.exit = deque(maxlen=100)
        self.counted_enter = set()
        self.counted_exit = set()
        
        self.enter2 = deque(maxlen=100)
        self.exit2 = deque(maxlen=100)
        self.counted_enter2 = set()
        self.counted_exit2 = set()
        
        self.cards_given = 0
        
        # Define tracking areas
        self.areas = {
            'area1': np.array([(327, 292), (322, 328), (880, 328), (880, 292)], np.int32),
            'area2': np.array([(322, 336), (312, 372), (880, 372), (880, 336)], np.int32),
            'area3': np.array([(338, 78), (338, 98), (870, 98), (870, 78)], np.int32),
            'area4': np.array([(341, 107), (341, 130), (870, 130), (870, 107)], np.int32)
        }

    def process_frame(self, frame, frame_count, frame_skip):
        try: 
            frame = cv2.resize(frame, (1020, 600))
            results = model.track(frame, persist=True)
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None
                confidences = results[0].boxes.conf.cpu().numpy()
                
                current_tracks = set()

                for idx, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                    track_id = track_ids[idx] if track_ids is not None else idx
                    c = self.names[int(class_id)]
                    x1, y1, x2, y2 = box.astype(int)
                    point = (x1, y2)
                    current_tracks.add(track_id)

                    color = (0, 255, 0) if "Person" in c else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{c} {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Draw tracking point
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)

                    # Person tracking
                    if "Person" in c:
                        self.handle_person_movement(track_id, point)

                    # Employee tracking
                    if "P1" in c or "P2" in c:
                        self.handle_employee_movement(track_id, point, c)

                    # Card detection
                    if "Card" in c:
                        self.cards_given += 1

                self.cleanup_tracks(current_tracks)
                self.update_statistics()

                # Draw counting information
                cv2.putText(frame, f"Active People: {self.active_people}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Total Entered: {len(self.counted_enter)}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Total Exited: {len(self.counted_exit)}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Draw entry/exit zones with labels
                cv2.polylines(frame, [self.areas['area1']], True, (0, 255, 0), 2)  # Entry zone (green)
                cv2.putText(frame, "ENTRY", tuple(self.areas['area1'][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.polylines(frame, [self.areas['area2']], True, (0, 0, 255), 2)  # Exit zone (red)
                cv2.putText(frame, "EXIT", tuple(self.areas['area2'][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Show the frame
#                cv2.imshow("Tracking Visualization", frame)
 #               if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
  #                  raise KeyboardInterrupt("User stopped visualization")

        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}", exc_info=True)

    # def handle_person_movement(self, track_id, point):
    #     # Area1 -> Area2 = Enter
    #     if cv2.pointPolygonTest(self.areas['area1'], point, False) >= 0:
    #         self.enter.append(track_id)
    #
    #     if track_id in self.enter and cv2.pointPolygonTest(self.areas['area2'], point, False) >= 0:
    #         self.counted_enter.add(track_id)
    #         logger.info(f"Person {track_id} entered zone")
    #
    #     # Area2 -> Area1 = Exit
    #     if cv2.pointPolygonTest(self.areas['area2'], point, False) >= 0:
    #         self.exit.append(track_id)
    #
    #     if track_id in self.exit and cv2.pointPolygonTest(self.areas['area1'], point, False) >= 0:
    #         self.counted_exit.add(track_id)
    #         logger.info(f"Person {track_id} exited zone")
    #
    # def handle_employee_movement(self, track_id, point, employee_type):
    #     # Area3 -> Area4 = Enter
    #     if cv2.pointPolygonTest(self.areas['area3'], point, False) >= 0:
    #         self.enter2.append(track_id)
    #
    #     if track_id in self.enter2 and cv2.pointPolygonTest(self.areas['area4'], point, False) >= 0:
    #         self.counted_enter2.add(track_id)
    #         self.log_employee_entry(employee_type)
    #
    #     # Area4 -> Area3 = Exit
    #     if cv2.pointPolygonTest(self.areas['area4'], point, False) >= 0:
    #         self.exit2.append(track_id)
    #
    #     if track_id in self.exit2 and cv2.pointPolygonTest(self.areas['area3'], point, False) >= 0:
    #         self.counted_exit2.add(track_id)
    #         self.log_employee_exit(employee_type)

    def handle_person_movement(self, track_id, point):
        test_point = (int(point[0]), int(point[1]))
        in_entry = cv2.pointPolygonTest(self.areas['area1'], test_point, False) >= 0
        in_exit = cv2.pointPolygonTest(self.areas['area2'], test_point, False) >= 0

        # Entry logic
        if in_entry and track_id not in self.enter:
            self.enter.append(track_id)

        if track_id in self.enter and in_exit and track_id not in self.counted_enter:
            self.counted_enter.add(track_id)
            logger.info(f"Person {track_id} entered zone")

        # Exit logic
        if in_exit and track_id not in self.exit:
            self.exit.append(track_id)

        if track_id in self.exit and in_entry and track_id not in self.counted_exit:
            self.counted_exit.add(track_id)
            logger.info(f"Person {track_id} exited zone")


    def handle_employee_movement(self, track_id, point, employee_type):
        test_point = (int(point[0]), int(point[1]))

        # Area3 -> Area4 = Enter
        if cv2.pointPolygonTest(self.areas['area3'], test_point, False) >= 0:
            self.enter2.append(track_id)

        if track_id in self.enter2 and cv2.pointPolygonTest(self.areas['area4'], test_point, False) >= 0:
            self.counted_enter2.add(track_id)
            self.log_employee_entry(employee_type)

        # Area4 -> Area3 = Exit
        if cv2.pointPolygonTest(self.areas['area4'], test_point, False) >= 0:
            self.exit2.append(track_id)

        if track_id in self.exit2 and cv2.pointPolygonTest(self.areas['area3'], test_point, False) >= 0:
            self.counted_exit2.add(track_id)
            self.log_employee_exit(employee_type)



    def log_employee_entry(self, employee_type):
        emp_id = 31 if employee_type == 'P1' else 32
        self.send_log(emp_id, 1)

    def log_employee_exit(self, employee_type):
        emp_id = 31 if employee_type == 'P1' else 32
        self.send_log(emp_id, 2)

    def send_log(self, employee_id, log_type):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        api_url = "https://backai.mycashtest.com/apiAdmin/employee/create_log"
        params = {
            "employee_id": employee_id,
            "type": log_type,
            "date": current_time
        }
        try:
            response = requests.post(api_url, params=params, timeout=3)
            response.raise_for_status()
            logger.info(f"Log added for employee {employee_id}")
        except Exception as e:
            logger.error(f"Error adding log: {str(e)}")

    def cleanup_tracks(self, current_tracks):
        # Remove tracks that are no longer detected
        for track_id in list(self.counted_enter):
            if track_id not in current_tracks:
                self.counted_enter.discard(track_id)
        
        for track_id in list(self.counted_exit):
            if track_id not in current_tracks:
                self.counted_exit.discard(track_id)

    def update_statistics(self):
        with stats_lock:
            # Calculate active people (entered - exited)
            self.active_people = len(self.counted_enter) - len(self.counted_exit)
            self.entered_zone = len(self.counted_enter)

            # Ensure we don't have negative counts
            if self.active_people < 0:
                self.active_people = 0
                logger.warning("Negative people count detected, resetting to 0")

            stats = {
                "active_people": self.active_people,
                "entered_zone": self.entered_zone,
                "exited_zone": len(self.counted_exit)
            }

            try:
                db.child("statistics").set(stats)
                logger.debug(f"Updated stats: {stats}")
            except Exception as e:
                logger.error(f"Firebase update failed: {str(e)}")

# Initialize the tracker
tracker = PeopleTracker()

def process_frame(frame, frame_count, frame_skip):
    tracker.process_frame(frame, frame_count, frame_skip)

if __name__ == "__main__":
    from streamer import generate as start_streaming
    producer_thread = threading.Thread(target=start_streaming, daemon=True)
    producer_thread.start()
    app.run(host="0.0.0.0", port=5001, threaded=True)
