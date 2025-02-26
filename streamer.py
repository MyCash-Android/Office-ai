import cv2
import subprocess
import time
import numpy as np
from app import process_frame

rtsp_url = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100?tcp"

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-re",
    "-rtsp_transport",
    "tcp",
    "-i",
    rtsp_url,
    "-rtbufsize",
    "400M",
    "-vf",
    "scale=1280:720",
    "-r",
    "10",
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

ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
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
