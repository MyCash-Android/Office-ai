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


if __name__ == "__main__":
    try:
        generate()
    except KeyboardInterrupt:
        print("Streaming interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
