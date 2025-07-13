import streamlit as st
import tempfile
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

st.set_page_config(page_title="Football Video Processor", layout="centered")
st.title("⚽ Football Match Video Processor")

uploaded_file = st.file_uploader("🎥 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.success("✅ Video uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        input_path = temp_input_file.name

    # Show input video preview
    st.video(input_path)

    if st.button("🔄 Process Video"):
        with st.spinner("Processing video... Please wait."):
            try:
                video_frames = read_video(input_path)
                tracker = Tracker("models/best.pt")
                tracks = tracker.get_object_tracks(video_frames)
                tracker.add_position_to_tracks(tracks)
                tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

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

                player_assigner = PlayerBallAssigner()
                team_ball_control = []

                for frame_num, player_track in enumerate(tracks["players"]):
                    ball_info = tracks["ball"][frame_num]
                    ball_bbox = ball_info.get(1, {}).get("bbox", None)

                    if ball_bbox is None:
                        last = team_ball_control[-1] if frame_num > 0 else "None"
                        team_ball_control.append(last)
                        continue

                    assigned_player = player_assigner.assign_ball_to_player(
                        player_track, ball_bbox
                    )

                    if assigned_player != -1:
                        tracks["players"][frame_num][assigned_player]["has_ball"] = True
                        team = tracks["players"][frame_num][assigned_player].get(
                            "team", "None"
                        )
                        team_ball_control.append(team)
                    else:
                        last = team_ball_control[-1] if frame_num > 0 else "None"
                        team_ball_control.append(last)

                team_ball_control = np.array(team_ball_control)
                output_frames = tracker.draw_annotations(
                    video_frames, tracks, team_ball_control, team_assigner
                )

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_output_file:
                    output_path = temp_output_file.name
                    save_video(output_frames, output_path)

                # Final output: show message and download button only
                st.success("✅ Processing complete! Download your video below.")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇ Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )

            except Exception as e:
                st.error(f"❌ An error occurred during processing:\n\n{str(e)}")
