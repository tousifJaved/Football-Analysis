import streamlit as st
import tempfile
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

# Page config
st.set_page_config(page_title="Football Video Processor", layout="centered")
st.title("⚽ Football Match Video Processor")

# File upload
uploaded_file = st.file_uploader("🎥 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.success("✅ Video uploaded successfully!")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        input_path = temp_input_file.name

    st.video(input_path)

    if st.button("🔄 Process Video"):
        with st.spinner("Processing video... Please wait."):
            try:
                # Step 1: Read video frames
                video_frames = read_video(input_path)

                # Step 2: Track objects
                tracker = Tracker("models/best.pt")
                tracks = tracker.get_object_tracks(video_frames)

                # Step 3: Add object positions and interpolate ball
                tracker.add_position_to_tracks(tracks)
                tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

                # Step 4: Assign team colors
                team_assigner = TeamAssigner()
                team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

                for frame_num, player_track in enumerate(tracks["players"]):
                    for player_id, track in player_track.items():
                        team = team_assigner.get_player_team(
                            video_frames[frame_num], track["bbox"], player_id
                        )
                        track["team"] = team
                        track["team_color"] = team_assigner.team_colors.get(
                            team, (255, 255, 255)
                        )

                # Step 5: Assign ball possession
                player_assigner = PlayerBallAssigner()
                team_ball_control = []

                for frame_num, player_track in enumerate(tracks["players"]):
                    ball_info = tracks["ball"][frame_num]
                    ball_bbox = ball_info.get(1, {}).get("bbox", None)

                    if ball_bbox is None:
                        team_ball_control.append(
                            team_ball_control[-1] if frame_num > 0 else "None"
                        )
                        continue

                    assigned_player = player_assigner.assign_ball_to_player(
                        player_track, ball_bbox
                    )

                    if assigned_player != -1:
                        tracks["players"][frame_num][assigned_player]["has_ball"] = True
                        team_ball_control.append(
                            tracks["players"][frame_num][assigned_player]["team"]
                        )
                    else:
                        team_ball_control.append(
                            team_ball_control[-1] if frame_num > 0 else "None"
                        )

                team_ball_control = np.array(team_ball_control)

                # Step 6: Draw annotations
                output_frames = tracker.draw_annotations(
                    video_frames, tracks, team_ball_control
                )

                # Step 7: Save output to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".avi"
                ) as temp_output_file:
                    output_path = temp_output_file.name
                    save_video(output_frames, output_path)

                # Step 8: Display and download
                st.success("✅ Processing complete!")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇ Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.avi",
                        mime="video/avi",
                    )
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
