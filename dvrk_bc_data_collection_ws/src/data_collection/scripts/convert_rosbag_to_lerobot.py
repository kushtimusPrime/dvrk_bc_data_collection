#!/usr/bin/env python3
"""
Convert folder of rosbag files to LeRobot v2.1 dataset format.

This script processes a directory of .bag files and converts them to a complete
LeRobot v2.1 dataset with full metadata compatibility.

Processing steps:
1. Auto-discover all .bag files in input directory
2. For each bag: synchronize messages offline (nearest-neighbor matching)
3. Downsample to target framerate (default: 30 Hz)
4. Encode videos with PyAV/NVENC (H.264)
5. Create Parquet files with observation/action pairs
6. Generate metadata files (info.json, episodes.jsonl)

Usage:
    python convert_rosbag_to_lerobot.py \
        --input-dir data/raw_rosbags \
        --output data/lerobot_dataset \
        --fps 30

Arguments:
    --input-dir     Directory containing .bag files
    --output        Output LeRobot dataset directory
    --fps           Target framerate (default: 30)
    --slop          Sync tolerance in seconds (default: 0.1)
    --train-ratio   Train/test split ratio (default: 0.85)
"""

import rosbag
import numpy as np
from collections import defaultdict
import av
from cv_bridge import CvBridge
import argparse
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================================
# Video Encoding (copied from data_collection_node.py)
# ============================================================================

class VideoWriter:
    """Fast H.264 video writer using PyAV (FFmpeg bindings)"""

    def __init__(self, output_path, width, height, fps):
        self.container = av.open(output_path, mode='w')

        # Use hardware encoding (NVIDIA h264_nvenc - 10-20x faster)
        self.stream = self.container.add_stream('h264_nvenc', rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'

        # Additional optimizations for h264_nvenc
        self.stream.codec_context.options = {
            'preset': 'p1',  # Fastest preset for nvenc (p1-p7, p1 is fastest)
            'tune': 'ull',   # Ultra-low latency
        }

    def write_frame(self, img_bgr):
        """
        Write a single frame (BGR format from ROS/OpenCV)
        img_bgr: numpy array (H, W, 3) uint8
        """
        # Convert BGR to RGB
        img_rgb = img_bgr[:, :, ::-1]

        # Create video frame
        frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')

        # Encode and write
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        """Flush remaining packets and close file"""
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


# ============================================================================
# Offline Message Synchronization
# ============================================================================

def synchronize_bag_messages(bag_path, slop=0.1):
    """
    Read bag and extract synchronized frames using offline nearest-neighbor matching.

    Args:
        bag_path: Path to rosbag file
        slop: Maximum time difference (seconds) for matching messages

    Returns:
        List of synchronized frames, each containing:
        {
            'timestamp': float,
            'img_left': sensor_msgs/Image,
            'img_right': sensor_msgs/Image,
            'psm1_joint_pos': np.array,
            'psm1_jaw_pos': np.array,
            'psm2_joint_pos': np.array,
            'psm2_jaw_pos': np.array,
        }
    """
    bag = rosbag.Bag(bag_path)

    # Buffer messages by topic
    messages = defaultdict(list)
    topics = [
        "/av/img_left_rect",
        "/av/img_right_rect",
        "/dvrk/PSM1/state_joint_current",
        "/dvrk/PSM1/state_jaw_current",
        "/dvrk/PSM2/state_joint_current",
        "/dvrk/PSM2/state_jaw_current",
    ]

    print(f"  Reading bag: {bag_path}")
    for topic, msg, t in bag.read_messages(topics=topics):
        # Store (timestamp, message) tuples
        ts = msg.header.stamp.to_sec()
        messages[topic].append((ts, msg))

    bag.close()

    print(f"  Messages per topic:")
    for topic, msgs in messages.items():
        print(f"    {topic.split('/')[-1]}: {len(msgs)}")

    # Use left camera as reference (typically lowest rate)
    ref_msgs = messages["/av/img_left_rect"]

    if len(ref_msgs) == 0:
        raise ValueError("No messages found for /av/img_left_rect")

    print(f"  Synchronizing {len(ref_msgs)} reference frames...")

    synchronized_frames = []
    dropped_frames = 0

    for ref_ts, ref_msg in tqdm(ref_msgs, desc="  Syncing"):
        # Find nearest message for each other topic within slop window

        # Right camera
        right_match = find_nearest_message(messages["/av/img_right_rect"], ref_ts, slop)
        if right_match is None:
            dropped_frames += 1
            continue

        # PSM1 joint
        psm1_joint_match = find_nearest_message(messages["/dvrk/PSM1/state_joint_current"], ref_ts, slop)
        if psm1_joint_match is None:
            dropped_frames += 1
            continue

        # PSM1 jaw
        psm1_jaw_match = find_nearest_message(messages["/dvrk/PSM1/state_jaw_current"], ref_ts, slop)
        if psm1_jaw_match is None:
            dropped_frames += 1
            continue

        # PSM2 joint
        psm2_joint_match = find_nearest_message(messages["/dvrk/PSM2/state_joint_current"], ref_ts, slop)
        if psm2_joint_match is None:
            dropped_frames += 1
            continue

        # PSM2 jaw
        psm2_jaw_match = find_nearest_message(messages["/dvrk/PSM2/state_jaw_current"], ref_ts, slop)
        if psm2_jaw_match is None:
            dropped_frames += 1
            continue

        # All messages matched - create synchronized frame
        frame = {
            'timestamp': ref_ts,
            'img_left': ref_msg,
            'img_right': right_match[1],
            'psm1_joint_pos': np.array(psm1_joint_match[1].position[:6], dtype=np.float32),
            'psm1_jaw_pos': np.array([psm1_jaw_match[1].position[0]], dtype=np.float32),
            'psm2_joint_pos': np.array(psm2_joint_match[1].position[:6], dtype=np.float32),
            'psm2_jaw_pos': np.array([psm2_jaw_match[1].position[0]], dtype=np.float32),
        }
        synchronized_frames.append(frame)

    drop_rate = (dropped_frames / len(ref_msgs)) * 100 if len(ref_msgs) > 0 else 0
    print(f"  ✓ Synchronized {len(synchronized_frames)} frames (drop rate: {drop_rate:.2f}%)")

    if drop_rate > 5.0:
        print(f"  ⚠ WARNING: Drop rate >5% - consider increasing slop parameter")

    return synchronized_frames


def find_nearest_message(msg_list, target_ts, max_diff):
    """
    Find message with timestamp nearest to target_ts within max_diff window.

    Args:
        msg_list: List of (timestamp, message) tuples
        target_ts: Target timestamp
        max_diff: Maximum time difference allowed

    Returns:
        (timestamp, message) tuple, or None if no match within window
    """
    best_match = None
    best_diff = float('inf')

    for ts, msg in msg_list:
        diff = abs(ts - target_ts)
        if diff < best_diff and diff <= max_diff:
            best_diff = diff
            best_match = (ts, msg)

    return best_match


# ============================================================================
# Temporal Downsampling
# ============================================================================

def downsample_frames(frames, target_fps):
    """
    Downsample frames to target framerate by selecting frames at consistent intervals.

    Args:
        frames: List of synchronized frames (can be variable rate)
        target_fps: Target framerate (e.g., 30 Hz)

    Returns:
        List of downsampled frames at consistent target_fps
    """
    if len(frames) == 0:
        return []

    # Calculate actual framerate from timestamps
    timestamps = [f['timestamp'] for f in frames]
    duration = timestamps[-1] - timestamps[0]
    actual_fps = len(frames) / duration if duration > 0 else 0

    print(f"  Downsampling from {actual_fps:.1f} Hz to {target_fps} Hz")

    # If actual fps is already close to target, no downsampling needed
    if abs(actual_fps - target_fps) < 1.0:
        print(f"  ✓ Framerate already close to target ({actual_fps:.1f} Hz), skipping downsample")
        return frames

    # Calculate target time interval
    dt_target = 1.0 / target_fps

    # Select frames at regular intervals
    downsampled = []
    current_target_time = timestamps[0]

    for frame in frames:
        # If this frame is past the current target time, include it
        if frame['timestamp'] >= current_target_time:
            downsampled.append(frame)
            current_target_time += dt_target

    actual_downsampled_fps = len(downsampled) / duration if duration > 0 else 0
    print(f"  ✓ Downsampled to {len(downsampled)} frames ({actual_downsampled_fps:.1f} Hz)")

    return downsampled


# ============================================================================
# Video Encoding from Synchronized Frames
# ============================================================================

def encode_videos(frames, output_dir, episode_idx, fps=30):
    """
    Encode synchronized frames to MP4 videos.
    No time pressure - can take as long as needed.
    """
    video_dir = Path(output_dir) / "videos" / "chunk-000"
    left_dir = video_dir / "observation.images.left"
    right_dir = video_dir / "observation.images.right"

    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_video = VideoWriter(
        str(left_dir / f"episode_{episode_idx:06d}.mp4"),
        width=1280, height=960, fps=fps
    )
    right_video = VideoWriter(
        str(right_dir / f"episode_{episode_idx:06d}.mp4"),
        width=1280, height=960, fps=fps
    )

    bridge = CvBridge()

    print(f"  Encoding {len(frames)} frames to video...")
    for frame in tqdm(frames, desc="  Encoding"):
        img_left = bridge.imgmsg_to_cv2(frame['img_left'], "bgr8")
        img_right = bridge.imgmsg_to_cv2(frame['img_right'], "bgr8")

        left_video.write_frame(img_left)
        right_video.write_frame(img_right)

    left_video.close()
    right_video.close()

    print(f"  ✓ Videos encoded")


# ============================================================================
# LeRobot v2.1 Dataset Creation
# ============================================================================

def construct_state_vector(psm1_joints, psm1_jaw, psm2_joints, psm2_jaw):
    """
    Concatenate: [PSM1_j1...j6, PSM1_gripper, PSM2_j1...j6, PSM2_gripper]
    Returns: np.array of shape (14,)
    """
    return np.concatenate([
        psm1_joints,  # (6,)
        psm1_jaw,     # (1,)
        psm2_joints,  # (6,)
        psm2_jaw,     # (1,)
    ])


def construct_action_observation_pairs(frames):
    """
    Action at timestep t = state at timestep t+1 (temporal offset for BC)
    For the last frame, action = observation (no next state available)

    Args:
        frames: List of synchronized frames

    Returns:
        (observations, actions) tuple of numpy arrays
    """
    states = []
    for frame in frames:
        state = construct_state_vector(
            frame['psm1_joint_pos'],
            frame['psm1_jaw_pos'],
            frame['psm2_joint_pos'],
            frame['psm2_jaw_pos']
        )
        states.append(state)

    states_array = np.array(states)

    # Observations: all N frames
    observations = states_array

    # Actions: shifted by 1, last action repeats last observation
    actions = np.concatenate([states_array[1:], states_array[-1:]], axis=0)

    return observations, actions


def create_episode_parquet(frames, episode_idx, output_dir, fps=30, chunk_idx=0):
    """
    Create a single Parquet file for one episode (LeRobot v2.1 style)

    Args:
        frames: List of synchronized frames
        episode_idx: Episode number (0-indexed)
        output_dir: Base LeRobot dataset directory
        fps: Target framerate for timestamp calculations
        chunk_idx: Chunk index for organizing large datasets

    Returns:
        Number of frames in episode
    """
    # Construct state/action pairs
    observations, actions = construct_action_observation_pairs(frames)
    n_frames = len(observations)

    # Build data dictionary for this episode
    data_dict = {
        'observation.state': observations,
        'action': actions,
        'frame_index': list(range(n_frames)),
        'timestamp': [i / fps for i in range(n_frames)],  # Episode-relative time
        'next.done': [i == n_frames - 1 for i in range(n_frames)],

        # Video frame references (relative paths)
        'observation.images.left': [
            {
                'path': f'videos/chunk-{chunk_idx:03d}/observation.images.left/episode_{episode_idx:06d}.mp4',
                'timestamp': i / fps
            } for i in range(n_frames)
        ],
        'observation.images.right': [
            {
                'path': f'videos/chunk-{chunk_idx:03d}/observation.images.right/episode_{episode_idx:06d}.mp4',
                'timestamp': i / fps
            } for i in range(n_frames)
        ],
    }

    # Convert to PyArrow table
    table = pa.table({
        'observation.state': pa.array([obs.tolist() for obs in data_dict['observation.state']]),
        'action': pa.array([act.tolist() for act in data_dict['action']]),
        'frame_index': pa.array(data_dict['frame_index']),
        'timestamp': pa.array(data_dict['timestamp']),
        'next.done': pa.array(data_dict['next.done']),
        'observation.images.left': pa.array([str(v) for v in data_dict['observation.images.left']]),
        'observation.images.right': pa.array([str(v) for v in data_dict['observation.images.right']]),
    })

    # Write to Parquet file
    parquet_dir = Path(output_dir) / "data" / f"chunk-{chunk_idx:03d}"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = parquet_dir / f"episode_{episode_idx:06d}.parquet"
    pq.write_table(table, parquet_path, compression='snappy')

    print(f"  ✓ Parquet file created: {parquet_path}")

    return n_frames


# ============================================================================
# LeRobot v2.1 Dataset Metadata
# ============================================================================

def create_info_json(output_dir, num_episodes, fps=30, train_ratio=0.85):
    """
    Create info.json with dataset metadata for LeRobot v2.1.

    Args:
        output_dir: Output dataset directory
        num_episodes: Total number of episodes
        fps: Dataset framerate
        train_ratio: Ratio of episodes for training (default 0.85 = 85%)
    """
    # Calculate train/test split
    train_end = int(num_episodes * train_ratio)

    info = {
        'codebase_version': 'v2.1',
        'robot_type': 'dvrk',
        'fps': fps,
        'splits': {
            'train': f'0:{train_end}',
            'test': f'{train_end}:{num_episodes}',
        },
        'data_path': 'data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{chunk_idx:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'features': {
            'observation.state': {
                'dtype': 'float32',
                'shape': [14],
                'names': [
                    'psm1_outer_yaw', 'psm1_outer_pitch', 'psm1_outer_insertion',
                    'psm1_outer_roll', 'psm1_outer_wrist_pitch', 'psm1_outer_wrist_yaw',
                    'psm1_jaw',
                    'psm2_outer_yaw', 'psm2_outer_pitch', 'psm2_outer_insertion',
                    'psm2_outer_roll', 'psm2_outer_wrist_pitch', 'psm2_outer_wrist_yaw',
                    'psm2_jaw',
                ]
            },
            'action': {
                'dtype': 'float32',
                'shape': [14],
                'names': [
                    'psm1_outer_yaw', 'psm1_outer_pitch', 'psm1_outer_insertion',
                    'psm1_outer_roll', 'psm1_outer_wrist_pitch', 'psm1_outer_wrist_yaw',
                    'psm1_jaw',
                    'psm2_outer_yaw', 'psm2_outer_pitch', 'psm2_outer_insertion',
                    'psm2_outer_roll', 'psm2_outer_wrist_pitch', 'psm2_outer_wrist_yaw',
                    'psm2_jaw',
                ]
            },
            'observation.images.left': {'dtype': 'video'},
            'observation.images.right': {'dtype': 'video'},
            'frame_index': {'dtype': 'int64'},
            'timestamp': {'dtype': 'float32'},
            'next.done': {'dtype': 'bool'},
        }
    }

    info_path = Path(output_dir) / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✓ Created info.json: {info_path}")


def create_episodes_jsonl(episodes_metadata, output_dir):
    """
    Create episodes.jsonl with per-episode metadata.

    Args:
        episodes_metadata: List of episode metadata dictionaries
        output_dir: Output dataset directory
    """
    jsonl_path = Path(output_dir) / "meta" / "episodes.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, 'w') as f:
        for ep_meta in episodes_metadata:
            # Write each episode as a JSON line
            json_line = json.dumps({
                'episode_index': ep_meta['episode_index'],
                'tasks': 'bimanual_manipulation',
                'length': ep_meta['num_frames'],
                'duration_sec': ep_meta.get('duration_sec', 0),
            })
            f.write(json_line + '\n')

    print(f"✓ Created episodes.jsonl: {jsonl_path}")


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_episode(bag_path, output_dir, episode_idx, slop=0.1, fps=30):
    """
    Main processing pipeline for one episode.

    Args:
        bag_path: Path to rosbag file
        output_dir: Output LeRobot dataset directory
        episode_idx: Episode index (0-based)
        slop: Time synchronization tolerance (seconds)
        fps: Target framerate for output dataset (default: 30)
    """
    print(f"\n{'='*70}")
    print(f"Processing episode {episode_idx+1}: {Path(bag_path).name}")
    print(f"{'='*70}")

    # Step 1: Synchronize messages
    frames = synchronize_bag_messages(bag_path, slop=slop)

    if len(frames) == 0:
        raise ValueError("No frames synchronized - check bag file and slop parameter")

    # Step 2: Downsample to target framerate
    frames = downsample_frames(frames, target_fps=fps)

    if len(frames) == 0:
        raise ValueError("No frames after downsampling")

    # Step 3: Encode videos
    encode_videos(frames, output_dir, episode_idx, fps=fps)

    # Step 4: Create Parquet
    n_frames = create_episode_parquet(frames, episode_idx, output_dir, fps=fps)

    # Calculate episode statistics
    timestamps = [f['timestamp'] for f in frames]
    duration = timestamps[-1] - timestamps[0]
    actual_fps = n_frames / duration if duration > 0 else 0

    print(f"\n  Episode {episode_idx} summary:")
    print(f"    Frames: {n_frames}")
    print(f"    Duration: {duration:.2f}s")
    print(f"    Actual FPS: {actual_fps:.1f}")
    print(f"    Target FPS: {fps}")
    print(f"  ✓ Episode complete\n")

    return {
        'episode_index': episode_idx,
        'num_frames': n_frames,
        'duration_sec': duration,
        'fps': actual_fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert folder of rosbags to LeRobot v2.1 dataset format'
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .bag files")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--slop", type=float, default=0.1,
                       help="Synchronization slop in seconds (default: 0.1)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target framerate for output dataset (default: 30)")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                       help="Train/test split ratio (default: 0.85)")

    args = parser.parse_args()

    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_path}")
        return 1

    # Find all .bag files
    bag_files = sorted(input_path.glob("*.bag"))
    if len(bag_files) == 0:
        print(f"ERROR: No .bag files found in {input_path}")
        return 1

    num_episodes = len(bag_files)

    print(f"\n{'='*70}")
    print(f"Converting rosbags to LeRobot v2.1 dataset")
    print(f"{'='*70}")
    print(f"  Input directory: {input_path}")
    print(f"  Output directory: {args.output}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"{'='*70}\n")

    # Process each bag file
    episodes_metadata = []
    for episode_idx, bag_path in enumerate(bag_files):
        try:
            episode_meta = process_episode(
                str(bag_path),
                args.output,
                episode_idx,
                slop=args.slop,
                fps=args.fps
            )

            # Save episode metadata
            meta_dir = Path(args.output) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)

            meta_file = meta_dir / f"episode_{episode_idx:06d}_meta.json"
            with open(meta_file, 'w') as f:
                json.dump(episode_meta, f, indent=2)

            # Collect metadata for finalization
            episodes_metadata.append(episode_meta)

        except Exception as e:
            print(f"\nERROR processing episode {episode_idx} ({bag_path.name}): {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Finalize dataset
    print(f"\n{'='*70}")
    print(f"Finalizing LeRobot dataset")
    print(f"{'='*70}\n")

    try:
        create_info_json(args.output, num_episodes, fps=args.fps, train_ratio=args.train_ratio)
        create_episodes_jsonl(episodes_metadata, args.output)

        print(f"\n{'='*70}")
        print(f"✓ Dataset conversion complete")
        print(f"{'='*70}")
        print(f"  Output: {args.output}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Status: Ready for LeRobot compatibility")
        print(f"{'='*70}\n")

        return 0

    except Exception as e:
        print(f"\nERROR during finalization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
