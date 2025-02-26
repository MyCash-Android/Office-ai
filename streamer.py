import cv2
import subprocess
import time
import numpy as np
from app import process_frame

rtsp_url = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100?tcp"

def generate():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("ERROR: Unable to open RTSP stream. Check camera URL.")
        return
    frame_count=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting to RTSP stream...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        process_frame(frame, frame_count=frame_count, frame_skip=1)
        frame_count += 1

if __name__ == "__main__":
    try:
        generate()
    except KeyboardInterrupt:
        print("Streaming interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
