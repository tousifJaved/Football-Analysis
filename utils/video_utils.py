import cv2
import tempfile
import os


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=24, codec="mp4v"):
    if not output_video_frames:
        raise ValueError("No frames provided.")
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def encode_video_to_bytes(frames, fps=15, codec="XVID"):
    if not frames:
        raise ValueError("No frames to encode.")
    height, width = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp_file:
        temp_path = tmp_file.name
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    with open(temp_path, "rb") as f:
        video_bytes = f.read()
    os.remove(temp_path)
    return video_bytes
