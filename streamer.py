import cv2
import subprocess
import time
from app import process_frame, update_latest_frame

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
    "1500k",
    "-pix_fmt",
    "yuv420p",
    "-flags",
    "+low_delay",
    "-flags",
    "+error_resilient",
    "-err_detect",
    "ignore_err",
    "-fflags",
    "+nobuffer",
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


def generate():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("ERROR: Unable to open RTSP stream. Check camera URL.")
        return

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting to RTSP stream...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        processed_frame = process_frame(frame, frame_count=0, frame_skip=1)
        if processed_frame is None:
            continue

        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2YUV)

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
