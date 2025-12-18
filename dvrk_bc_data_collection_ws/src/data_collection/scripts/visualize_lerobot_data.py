"""
Visualize dVRK LeRobot v2.1 dataset.

This script visualizes robot trajectories and camera views from a Le Robot dataset,
showing both observation (current state) and action (next state) overlaid.

Usage:
    python visualize_lerobot_data.py data/initial_test_lerobot
"""
from torchcodec.decoders import VideoDecoder
import tyro
import viser
from viser.extras import ViserUrdf
import numpy as np
from pathlib import Path
import pandas as pd
import time
import json
import ast

DEVICE = "cpu"


def main(dataset_dir: str):
    """
    Visualize dVRK LeRobot dataset.

    Args:
        dataset_dir: Path to LeRobot dataset directory (e.g., data/initial_test_lerobot)
    """
    server = viser.ViserServer()
    current_left_video_path = None
    current_right_video_path = None
    frame_index = 0
    joints_data = None

    # Load URDF for dVRK robots
    urdf_path = Path(__file__).parent.parent.parent.parent.parent / "dvrk_bc_data_collection" / "dvrk_description" / "psm1_and_psm2.urdf"

    # Observation robot (current state)
    observation_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=True,
        load_collision_meshes=False,
        root_node_name="/observation_robot",
    )

    # Action robot (next state) - same position as observation
    action_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=True,
        mesh_color_override=(0, 1, 0),
        load_collision_meshes=False,
        root_node_name="/action_robot",
    )

    # Get number of episodes by counting Parquet files
    dataset_path = Path(dataset_dir)
    data_dir = dataset_path / "data" / "chunk-000"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"Expected structure: {dataset_dir}/data/chunk-000/episode_*.parquet")
        return

    # Count episode Parquet files
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    num_episodes = len(parquet_files)

    if num_episodes == 0:
        print(f"ERROR: No episode Parquet files found in {data_dir}")
        return

    print(f"Found {num_episodes} episodes in dataset")

    # GUI controls
    episode_options = [f"Episode {i:06d}" for i in range(num_episodes)]
    episode_dropdown = server.gui.add_dropdown(
        "Select episode",
        options=episode_options,
        initial_value=episode_options[0]
    )

    # Robot visibility controls
    with server.gui.add_folder("Robot Visibility"):
        show_observation = server.gui.add_checkbox("Show Observation Robot", initial_value=True)
        show_action = server.gui.add_checkbox("Show Action Robot", initial_value=True)

        @show_observation.on_update
        def _(_):
            observation_urdf.show_visual = show_observation.value

        @show_action.on_update
        def _(_):
            action_urdf.show_visual = show_action.value

    # Image handles for left and right cameras
    left_image_handle = server.gui.add_image(
        image=np.zeros((100,100,3)),
        label="Left Camera",
        visible=False,
        format='jpeg'
    )
    right_image_handle = server.gui.add_image(
        image=np.zeros((100,100,3)),
        label="Right Camera",
        visible=False,
        format='jpeg'
    )

    # Frame index slider and controls
    frame_slider = server.gui.add_slider("Frame", 0, 1, step=1, initial_value=0)
    frame_slider.on_update(lambda _: show_frame(frame_slider.value))

    left_video = None
    right_video = None
    left_video_data = None
    right_video_data = None
    loaded_sequence = False

    def load_sequence(episode_idx):
        """Load episode data from LeRobot v2.1 Parquet format"""
        nonlocal current_left_video_path, current_right_video_path, frame_index
        nonlocal left_video, right_video, left_video_data, right_video_data
        nonlocal loaded_sequence, joints_data

        print(f"Loading episode {episode_idx}...")
        loaded_sequence = False

        # Load Parquet file
        parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        if not parquet_path.exists():
            print(f"ERROR: Parquet file not found: {parquet_path}")
            return

        df = pd.read_parquet(parquet_path)

        # Extract data - arrays are stored as strings, need to parse them
        observations = np.array([s for s in df['observation.state']])  # Shape: (N, 14)
        actions = np.array([s for s in df['action']])  # Shape: (N, 14)

        print(f"  Loaded {len(observations)} frames")
        print(f"  Observation shape: {observations.shape}")
        print(f"  Action shape: {actions.shape}")

        # Video paths from first row
        left_video_ref = ast.literal_eval(df['observation.images.left'].iloc[0])
        right_video_ref = ast.literal_eval(df['observation.images.right'].iloc[0])

        current_left_video_path = dataset_path / left_video_ref['path']
        current_right_video_path = dataset_path / right_video_ref['path']

        print(f"  Left video: {current_left_video_path}")
        print(f"  Right video: {current_right_video_path}")

        # Load videos
        left_video = VideoDecoder(str(current_left_video_path), device=DEVICE)
        right_video = VideoDecoder(str(current_right_video_path), device=DEVICE)

        # Store data
        joints_data = {
            'observation': observations,  # (N, 14)
            'action': actions,            # (N, 14)
        }

        # Verify video and joint data lengths match
        assert len(left_video) == len(right_video) == len(observations), \
            f"Video and joint data lengths don't match: {len(left_video)} != {len(right_video)} != {len(observations)}"

        # Load all video frames
        frame_count_list = np.arange(len(left_video))
        left_video_data = left_video.get_frames_at(frame_count_list).data
        right_video_data = right_video.get_frames_at(frame_count_list).data

        loaded_sequence = True

        # Update slider
        frame_slider.min = 0
        frame_slider.max = len(observations) - 1
        frame_slider.value = 0

        print(f"✓ Loaded episode {episode_idx}")

    def show_frame(index: int):
        """Show a specific frame of data and video"""
        nonlocal left_image_handle, right_image_handle

        if not loaded_sequence or joints_data is None:
            return
        if index < 0 or index >= len(joints_data['observation']):
            return

        # Update videos
        left_frame = np.copy(left_video_data[index].permute(1, 2, 0).cpu().numpy())
        right_frame = np.copy(right_video_data[index].permute(1, 2, 0).cpu().numpy())

        left_image_handle.image = left_frame
        left_image_handle.visible = True
        right_image_handle.image = right_frame
        right_image_handle.visible = True

        # Update observation robot (current state)
        observation_joints = joints_data['observation'][index]  # Shape: (14,)
        observation_urdf.update_cfg(observation_joints)

        # Update action robot (next state)
        action_joints = joints_data['action'][index]  # Shape: (14,)
        # TODO: Mess around with jaws for visualization
        breakpoint()
        action_urdf.update_cfg(action_joints)

    # Load the first episode on startup
    initial_episode_idx = 0
    load_sequence(episode_idx=initial_episode_idx)

    @episode_dropdown.on_update
    def _(_) -> None:
        episode_idx = int(episode_dropdown.value.split()[1])
        load_sequence(episode_idx=episode_idx)

    playing = server.gui.add_checkbox("Playing", initial_value=False)

    while True:
        time.sleep(0.05)
        if playing.value and loaded_sequence:
            frame_slider.value = (frame_slider.value + 1) % len(left_video)


if __name__ == "__main__":
    tyro.cli(main)
