# DVRK BC Data Collection
Data Collection for the Open H Embodiment Project

## Installation
```bash
mamba create -n dvrk_bc_data_collection_env -c conda-forge -c robostack-noetic ros-noetic-desktop
mamba activate dvrk_bc_data_collection_env
mamba config --env --add channels robostack-noetic
mamba install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
mamba install -c conda-forge evdev
mamba install -y -c conda-forge av
cd ~/dvrk_bc_data_collection
pip install -e .
```

## Data Collection Pipeline

### 1. Record Rosbag Data

Start the rosbag collection node with keyboard controls:

```bash
cd dvrk_bc_data_collection_ws
catkin_make
source devel/setup.bash
rosrun data_collection rosbag_collection_node.py
```

**Keyboard Controls:**
- `Space` - Start recording an episode
- `Space` (again) - Stop recording the current episode
- `q` - Quit the data collection node

Rosbag files will be saved to `data/` directory with automatic episode numbering (e.g., `episode_0001.bag`, `episode_0002.bag`, etc.).

**Recorded Topics:**
- `/av/img_left_rect` - Left camera images (1280x960)
- `/av/img_right_rect` - Right camera images (1280x960)
- `/dvrk/PSM1/state_joint_current` - PSM1 joint positions
- `/dvrk/PSM1/state_jaw_current` - PSM1 jaw position
- `/dvrk/PSM2/state_joint_current` - PSM2 joint positions
- `/dvrk/PSM2/state_jaw_current` - PSM2 jaw position

Rosbags are automatically compressed with BZ2 compression to reduce file size.

---

### 2. Convert Rosbags to LeRobot Dataset

Convert a folder of rosbag files to LeRobot v2.1 format:

```bash
python src/data_collection/scripts/convert_rosbag_to_lerobot.py \
    --input-dir data/raw_rosbags \
    --output data/raw_rosbags_lerobot \
    --fps 30
```

**Arguments:**
- `--input-dir` - Directory containing `.bag` files
- `--output` - Output directory for LeRobot dataset
- `--fps` - Target framerate (default: 30 Hz)
- `--slop` - Synchronization tolerance in seconds (default: 0.1)
- `--train-ratio` - Train/test split ratio (default: 0.85)

**What this does:**
1. Auto-discovers all `.bag` files in the input directory
2. Synchronizes messages offline (no real-time pressure)
3. Downsamples to target framerate (e.g., 55 Hz → 30 Hz)
4. Encodes videos with H.264 (PyAV/NVENC)
5. Creates Parquet files with observation/action pairs
6. Generates metadata files (info.json, episodes.jsonl) for LeRobot compatibility

**Output structure:**
```
data/lerobot_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.left/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── observation.images.right/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episode_000000_meta.json
    └── ...
```

---

### 3. Visualize LeRobot Dataset

Visualize the collected dataset with 3D robot visualization and camera views:

```bash
python src/data_collection/scripts/visualize_lerobot_data.py data/lerobot_dataset
```

**Features:**
- 3D visualization of PSM1 and PSM2 robots using dVRK URDF
- Dual robot display:
  - **Observation robot** (current state) - default color
  - **Action robot** (next state) - green color
- Left and right camera views
- Episode selection dropdown
- Frame-by-frame scrubbing with slider
- Auto-play mode
- Toggle robot visibility

**Controls:**
- Use the **episode dropdown** to switch between episodes
- Use the **frame slider** to scrub through frames
- Check **Playing** to auto-play through frames
- Toggle **Show Observation Robot** / **Show Action Robot** to hide/show robots

The visualization will open a web interface (default: http://localhost:8080) where you can interactively explore the dataset.

---

## Data Format

The dataset uses **LeRobot v2.1 format** with the following state representation:

**State Vector (14 DOF):**
- Indices 0-5: PSM1 joints (outer_yaw, outer_pitch, outer_insertion, outer_roll, outer_wrist_pitch, outer_wrist_yaw)
- Index 6: PSM1 jaw
- Indices 7-12: PSM2 joints (outer_yaw, outer_pitch, outer_insertion, outer_roll, outer_wrist_pitch, outer_wrist_yaw)
- Index 13: PSM2 jaw

**Observation-Action Pairs:**
- `observation.state[t]` - Current robot state at frame t
- `action[t]` - Next robot state (= `observation.state[t+1]`)
- For the last frame, `action[N-1] = observation.state[N-1]` (no next state available)

---

## Troubleshooting

**Issue: Video/joint data length mismatch in visualization**
- Ensure you're using the latest version of `convert_rosbag_to_lerobot.py` which creates N observation-action pairs for N video frames

**Issue: Jaw not displaying correctly**
- Check that jaw values in rosbags are within URDF limits [0, 1.5707]
- Negative jaw values may indicate calibration issues

**Issue: High CPU usage during recording**
- This is expected with real-time video encoding; the rosbag approach offloads encoding to post-processing
- Rosbags use BZ2 compression to minimize file size during recording