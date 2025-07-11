import cv2
import tempfile


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


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # .avi format
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def encode_video_to_bytes(frames, fps=24):
    height, width, _ = frames[0].shape

    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_file:
        temp_path = tmp_file.name

    # Write video to temp path
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    # Read it back as bytes
    with open(temp_path, "rb") as f:
        video_bytes = f.read()

    return video_bytes
