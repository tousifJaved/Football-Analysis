from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_name == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions_extracted = [
            x.get(1, {}).get("bbox", []) for x in ball_positions
        ]
        valid_positions = [pos for pos in ball_positions_extracted if len(pos) == 4]

        if not valid_positions:
            return [{} for _ in range(len(ball_positions))]

        filled_positions = []
        for pos in ball_positions_extracted:
            if len(pos) == 4:
                filled_positions.append(pos)
            else:
                filled_positions.append([np.nan] * 4)

        df = pd.DataFrame(filled_positions, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += batch
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper class to player
            for i, cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            tracked_detections = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for det in tracked_detections:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = det[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for det in detection_supervision:
                bbox = det[0].tolist()
                cls_id = det[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1 = x_center - rect_w // 2
            x2 = x_center + rect_w // 2
            y1 = y2 - rect_h // 2 + 15
            y2_rect = y2 + rect_h // 2 + 15

            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2_rect)), color, cv2.FILLED
            )

            text_x = x1 + 12
            if track_id > 99:
                text_x -= 10

            cv2.putText(
                frame,
                str(track_id),
                (int(text_x), int(y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        h, w, _ = frame.shape

        # Dynamic box position based on frame size
        x1, y1 = int(0.7 * w), int(0.85 * h)
        x2, y2 = int(0.98 * w), int(0.98 * h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        control = team_ball_control[: frame_num + 1]
        t1 = (control == 1).sum()
        t2 = (control == 2).sum()
        total = t1 + t2 if (t1 + t2) > 0 else 1

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {t1 / total * 100:.2f}%",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {t2 / total * 100:.2f}%",
            (x1 + 10, y1 + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames
