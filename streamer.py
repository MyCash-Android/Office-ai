import cv2
import subprocess
import time
from app import process_frame, update_latest_frame  

rtsp_url = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100?tcp"

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "1020x600",
    "-r", "15",
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-b:v", "1000k",
    "-f", "flv",
    "rtmp://localhost/live/office",
    "-f", "hls",
    "-hls_time", "2",
    "-hls_list_size", "10",
    "-hls_flags", "delete_segments",
    "/var/www/html/hls/office.m3u8",
]

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
        processed_frame = process_frame(frame, frame_count, frame_skip)
        if processed_frame is None:
            continue

        try:
            ffmpeg_process.stdin.write(processed_frame.tobytes())
        except Exception as e:
            print("Error writing to ffmpeg:", e)

        update_latest_frame(processed_frame)
        
if __name__ == "__main__":
    try:
        generate()
    except KeyboardInterrupt:
        print("Streaming interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
