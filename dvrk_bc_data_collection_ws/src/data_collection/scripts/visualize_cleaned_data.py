import tyro
import viser
from viser.extras import ViserUrdf
import numpy as np
from pathlib import Path
import h5py
from torchcodec.decoders import VideoDecoder
import torch
import time
import tempfile
import subprocess
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_downsampled_video(input_path: str, scale_factor: float = 0.25) -> str:
    """
    Create a downsampled version of video using ffmpeg.
    Returns path to temporary downsampled file.
    """
    # Create temp file in same directory to avoid cross-filesystem issues
    input_dir = Path(input_path).parent
    temp_fd, temp_path = tempfile.mkstemp(suffix='_downsampled.mp4', dir=input_dir)
    os.close(temp_fd)

    # Calculate output resolution
    # For 5760×2880 with scale=0.25 → 1440×720
    scale_filter = f"scale=iw*{scale_factor}:ih*{scale_factor}"

    # Use ffmpeg to downsample
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', input_path,
        '-vf', scale_filter,
        '-c:v', 'libx264',
        '-preset', 'fast',  # Faster encoding
        '-crf', '23',  # Good quality
        '-pix_fmt', 'yuv420p',
        temp_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return temp_path


def main(data_dir: str):
    server = viser.ViserServer()
    current_wrist_left_video_path = None
    current_wrist_right_video_path = None
    current_exo_video_path = None
    current_insta360_video_path = None
    insta360_downsampled_path = None
    frame_index = 0
    joints_data = None

    # Load URDF for YAM robot (robot positions)
    urdf_path = Path(__file__).parent.parent.parent / "urdf" / "yam_bimanual.urdf"
    yam_viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=True,
        load_collision_meshes=False,
        root_node_name="/yam_robot",
    )

    # Get the min and max joint limits of URDF joints that contain "gripper"
    gripper_limits = {}
    gripper_joint_indices = []

    for idx, (joint_name, (lower, upper)) in enumerate(yam_viser_urdf.get_actuated_joint_limits().items()):
        if "gripper" in joint_name.lower():
            gripper_limits[joint_name] = {'lower': lower, 'upper': upper}
            gripper_joint_indices.append(idx)

    # Create a base frame for GELLOs that are 75cm behind the YAM robot
    server.scene.add_frame(
        "/gello_base",
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([-0.75, 0.0, 0.0]),
        show_axes=False,
    )

    # Load URDF for GELLO robot (human operator positions) as child of the base frame
    gello_viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=True,
        load_collision_meshes=False,
        root_node_name="/gello_base/gello_robot",
    )

    # Get list of all sorted folders in data_dir
    data_path = Path(data_dir)
    all_folders = sorted([f.name for f in data_path.iterdir() if f.is_dir()])

    # GUI controls
    trajectory_dropdown = server.gui.add_dropdown("Select a trajectory",options=all_folders,initial_value=all_folders[0])

    # Robot visibility controls
    with server.gui.add_folder("Robot Visibility"):
        show_yam_robot = server.gui.add_checkbox("Show YAM Robot (actual)", initial_value=True)
        show_gello_robot = server.gui.add_checkbox("Show GELLO Robot (operator)", initial_value=True)

        @show_yam_robot.on_update
        def _(_):
            yam_viser_urdf.show_visual = show_yam_robot.value

        @show_gello_robot.on_update
        def _(_):
            gello_viser_urdf.show_visual = show_gello_robot.value

    # Image handles for the wrist left, wrist right, and exo videos
    insta360_image_handle = server.gui.add_image(image=np.zeros((100,100,3)),label="Insta360 Image",visible=False,format='jpeg')
    exo_image_handle = server.gui.add_image(image=np.zeros((100,100,3)),label="Exo Image",visible=False,format='jpeg')
    wrist_left_image_handle = server.gui.add_image(image=np.zeros((100,100,3)),label="Wrist Left Image",visible=False,format='jpeg')
    wrist_right_image_handle = server.gui.add_image(image=np.zeros((100,100,3)),label="Wrist Right Image",visible=False,format='jpeg')
    

    # Frame index slider and controls
    frame_slider = server.gui.add_slider("Frame", 0, 1, step=1, initial_value=0)
    frame_slider.on_update(lambda _: show_frame(frame_slider.value))
    insta360_video = None
    wrist_left_video = None
    wrist_right_video = None
    exo_video = None
    insta360_video_data = None
    wrist_left_video_data = None
    wrist_right_video_data = None
    exo_video_data = None
    loaded_sequence = False

    def load_sequence(trajectory_dir):
        """Load the data and video from the processed directory"""
        nonlocal current_wrist_left_video_path, current_wrist_right_video_path, current_exo_video_path, current_insta360_video_path, insta360_downsampled_path, frame_index, wrist_left_video, wrist_right_video, exo_video, insta360_video, wrist_left_video_data, wrist_right_video_data, exo_video_data, insta360_video_data, loaded_sequence, joints_data
        print("Loading sequence ...")
        loaded_sequence = False

        # Clean up old downsampled video
        if insta360_downsampled_path and Path(insta360_downsampled_path).exists():
            try:
                Path(insta360_downsampled_path).unlink()
                print(f"Cleaned up old downsampled video: {insta360_downsampled_path}")
            except Exception as e:
                print(f"Warning: Could not delete old downsampled video: {e}")
            insta360_downsampled_path = None

        # Find the h5 file and video file
        current_wrist_left_video_path = f"{trajectory_dir}/wrist_left_spliced.mp4"
        current_wrist_right_video_path = f"{trajectory_dir}/wrist_right_spliced.mp4"
        current_exo_video_path = f"{trajectory_dir}/exo_spliced.mp4"
        current_insta360_video_path = f"{trajectory_dir}/insta360_spliced.mp4"

        # Open video (use CPU decoding to avoid CUDA format issues)
        wrist_left_video = VideoDecoder(current_wrist_left_video_path)
        wrist_right_video = VideoDecoder(current_wrist_right_video_path)
        exo_video = VideoDecoder(current_exo_video_path)

        # Create downsampled version of Insta360 video if it exists
        if Path(current_insta360_video_path).exists():
            print("Creating downsampled Insta360 video...")
            insta360_downsampled_path = create_downsampled_video(
                current_insta360_video_path,
                scale_factor=0.25
            )
            insta360_video = VideoDecoder(insta360_downsampled_path)
            print(f"Downsampled video created: {insta360_downsampled_path}")

        # Load joint data
        joints_h5_path = Path(trajectory_dir) / "joints.h5"
        if joints_h5_path.exists():
            with h5py.File(joints_h5_path, 'r') as f:
                joints_data = {
                    'yam_position': f['yam_position'][:],
                    'gello_position': f['gello_position'][:]
                }
            print(f"Loaded joints data: yam shape={joints_data['yam_position'].shape}, gello shape={joints_data['gello_position'].shape}")
        else:
            print(f"Warning: No joints.h5 found at {joints_h5_path}")
            joints_data = None
        frame_count_list = np.arange(len(wrist_left_video))
        if Path(current_insta360_video_path).exists():
            assert len(insta360_video) == len(wrist_left_video) == len(wrist_right_video) == len(exo_video) == len(joints_data['yam_position']) == len(joints_data['gello_position']), f"Insta360 video, Wrist left video, Wrist right video, Exo Video, YAM robot, and GELLO robot have different lengths: {len(insta360_video)} != {len(wrist_left_video)} != {len(wrist_right_video)} != {len(exo_video)} != {len(joints_data['yam_position'])} != {len(joints_data['gello_position'])}"
        else:
            assert len(wrist_left_video) == len(wrist_right_video) == len(exo_video) == len(joints_data['yam_position']) == len(joints_data['gello_position']), f"Wrist left video, Wrist right video, Exo Video, YAM robot, and GELLO robot have different lengths: {len(wrist_left_video)} != {len(wrist_right_video)} != {len(exo_video)} != {len(joints_data['yam_position'])} != {len(joints_data['gello_position'])}"

        wrist_left_video_data = wrist_left_video.get_frames_at(frame_count_list).data
        wrist_right_video_data = wrist_right_video.get_frames_at(frame_count_list).data
        exo_video_data = exo_video.get_frames_at(frame_count_list).data
        if Path(current_insta360_video_path).exists():
            insta360_video_data = insta360_video.get_frames_at(frame_count_list).data

        loaded_sequence = True
        # set slider min and max
        frame_slider.min = 0
        frame_slider.max = len(wrist_left_video_data) - 1
        frame_slider.value = 0
        print("Loaded sequence")
    
    def show_frame(index: int):
        nonlocal wrist_left_image_handle, wrist_right_image_handle, exo_image_handle, insta360_image_handle, insta360_video
        """Show a specific frame of data and video"""
        if not loaded_sequence or wrist_left_video_data is None:
            return
        if index < 0 or index >= len(wrist_left_video_data):
            return

        wrist_left_frame = np.copy(wrist_left_video_data[index].permute(1, 2, 0).cpu().numpy())
        wrist_right_frame = np.copy(wrist_right_video_data[index].permute(1, 2, 0).cpu().numpy())
        exo_frame = np.copy(exo_video_data[index].permute(1, 2, 0).cpu().numpy())
        if insta360_video_data:
            insta360_frame = np.copy(insta360_video_data[index].permute(1, 2, 0).cpu().numpy())
            insta360_image_handle.image = insta360_frame
            insta360_image_handle.visible = True

        
        wrist_left_image_handle.image = wrist_left_frame
        wrist_left_image_handle.visible = True
        wrist_right_image_handle.image = wrist_right_frame
        wrist_right_image_handle.visible = True
        exo_image_handle.image = exo_frame
        exo_image_handle.visible = True

        # Update URDF configurations
        if joints_data is not None and index < len(joints_data['yam_position']):
            # Update YAM robot (actual robot positions)
            yam_joints = joints_data['yam_position'][index].copy()

            # Scale the gripper joints to the gripper limits
            # Assumes input gripper values are in range [0, 1] and need to be scaled to [lower, upper]
            for gripper_idx in gripper_joint_indices:
                if gripper_idx < len(yam_joints):
                    # Get the first gripper limit (they should all be the same)
                    first_gripper_limit = list(gripper_limits.values())[0]
                    lower, upper = first_gripper_limit['lower'], first_gripper_limit['upper']

                    # Scale from [0, 1] to [lower, upper]
                    # If your data is in a different range, adjust accordingly
                    original_value = yam_joints[gripper_idx]
                    scaled_value = lower + original_value * (upper - lower)
                    yam_joints[gripper_idx] = scaled_value

            # Update the YAM URDF configuration
            yam_viser_urdf.update_cfg(yam_joints)

            # Update GELLO robot (human operator positions)
            if index < len(joints_data['gello_position']):
                gello_joints = joints_data['gello_position'][index].copy()

                # Scale the gripper joints for gello as well
                for gripper_idx in gripper_joint_indices:
                    if gripper_idx < len(gello_joints):
                        first_gripper_limit = list(gripper_limits.values())[0]
                        lower, upper = first_gripper_limit['lower'], first_gripper_limit['upper']
                        original_value = gello_joints[gripper_idx]
                        scaled_value = lower + original_value * (upper - lower)
                        gello_joints[gripper_idx] = scaled_value

                # Update the GELLO URDF configuration
                gello_viser_urdf.update_cfg(gello_joints)

    # Load the sequence on startup
    trajectory_path = data_path / trajectory_dropdown.value
    load_sequence(trajectory_dir=trajectory_path)

    @trajectory_dropdown.on_update
    def _(_) -> None:
        trajectory_path = data_path / trajectory_dropdown.value
        load_sequence(trajectory_dir=trajectory_path)

    playing = server.gui.add_checkbox("Playing", initial_value=False)
    while True:
        time.sleep(0.05)
        if playing.value and loaded_sequence:
            frame_slider.value = (frame_slider.value + 1) % len(wrist_left_video)

if __name__ == "__main__":
    tyro.cli(main)