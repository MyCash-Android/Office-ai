import queue
import cv2
import subprocess
import time
import numpy as np
import threading
from app import process_frame

q=queue.Queue()

rtsp_url = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100?tcp"
#rtsp_url = "testVid.mp4"

def receive():
    print("Start recieve")
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)

def generate():
    frame_count = 0
    frame_skip = 6
    print("Start generate")
    while True:
        if q.empty() != True:
            frame=q.get()
            if frame_count % frame_skip == 0:
                process_frame(frame, frame_count=frame_count, frame_skip=frame_skip)
                frame_count += 1
"""
def generate():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("ERROR: Unable to open RTSP stream. Check camera URL.")
        return

    frame_count = 0
    frame_skip = 1
    prev_frame = None  

    timestamp_x, timestamp_y, timestamp_width, timestamp_height = 0, 25, 400, 40  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting to RTSP stream...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(gray_frame, dtype=np.uint8) * 255  
        mask[timestamp_y:timestamp_y + timestamp_height, timestamp_x:timestamp_x + timestamp_width] = 0  

        masked_gray_frame = cv2.bitwise_and(gray_frame, mask)

        if prev_frame is not None:
            # Apply the same mask to the previous frame
            prev_masked_gray = cv2.bitwise_and(prev_frame, mask)

            # Compute absolute difference
            frame_diff = cv2.absdiff(masked_gray_frame, prev_masked_gray)
            _, thresh_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

            if cv2.countNonZero(thresh_diff) == 0:
                print("Duplicates detected (ignoring timestamp)")
                continue

        if frame_count % frame_skip == 0:
            process_frame(frame, frame_count=frame_count, frame_skip=frame_skip)

        prev_frame = masked_gray_frame.copy() 
        frame_count += 1
        time.sleep(0.05)
"""

if __name__ == "__main__":
    try:
        p1=threading.Thread(target=receive)
        p2=threading.Thread(target=generate)
        p1.start()
        p2.start()
    except KeyboardInterrupt:
        print("Streaming interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
