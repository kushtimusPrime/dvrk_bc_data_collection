import message_filters
from sensor_msgs.msg import Image, JointState
import rospy
from pynput import keyboard
import threading
import av
from cv_bridge import CvBridge
import time
import os
import numpy as np
import json
from datetime import datetime
import queue
import gc

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

class DataCollectionNode:
    def __init__(self):
        # Subscribe to all data streams
        self.img_left_sub = message_filters.Subscriber("/av/img_left", Image)
        self.img_right_sub = message_filters.Subscriber("/av/img_right", Image)
        self.psm1_joint_sub = message_filters.Subscriber("/dvrk/PSM1/state_joint_current", JointState)
        self.psm1_jaw_sub = message_filters.Subscriber("/dvrk/PSM1/state_jaw_current", JointState)
        self.psm2_joint_sub = message_filters.Subscriber("/dvrk/PSM2/state_joint_current", JointState)
        self.psm2_jaw_sub = message_filters.Subscriber("/dvrk/PSM2/state_jaw_current", JointState)

        self.video_writer_left = None
        self.video_writer_right = None
        self.data_dir = f"data/raw_{time.strftime('%Y%m%d_%H%M%S')}"
        self.cv_bridge = CvBridge()

        # Synchronize with larger queue and slop for mixed rates (cameras at 56Hz, joints at 99Hz)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.img_left_sub, self.img_right_sub, self.psm1_joint_sub, self.psm1_jaw_sub, self.psm2_joint_sub, self.psm2_jaw_sub],
            queue_size=50,  # Increased from 10 to handle rate differences
            slop=0.1  # Increased from 33ms to 100ms for better tolerance
        )
        sync.registerCallback(self.synchronized_callback)
        self.prev_time = None

        self.state = "IDLE"  # IDLE | RECORDING
        self.episode_buffer = []
        self.episode_count = 0
        self.state_lock = threading.Lock()

        # Encoding queue and background thread
        self.frame_queue = queue.Queue(maxsize=100)
        self.encoding_thread = threading.Thread(target=self._encoding_worker, daemon=True)
        self.encoding_thread.start()
        self.queue_warning_threshold = 80  # Warn when queue is 80% full

        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

        self.time_sync_debug = False

        # Log GC configuration
        gc_thresholds = gc.get_threshold()
        rospy.loginfo(f"🗑️  Python GC thresholds: gen0={gc_thresholds[0]} gen1={gc_thresholds[1]} gen2={gc_thresholds[2]}")
        rospy.loginfo(f"🗑️  Initial GC counts: {gc.get_count()}")

    def synchronized_callback(self, img_left, img_right, psm1_joint,
                             psm1_jaw, psm2_joint, psm2_jaw):
        """Called when all 6 messages are synchronized - queues data for background encoding"""
        t0_enter = time.perf_counter()

        # Check GC before callback (commented out - not logging)
        # gc_count_before = gc.get_count()

        # Extract timestamps
        t1 = time.perf_counter()
        timestamps = {
            'img_left': img_left.header.stamp.to_sec(),
            'img_right': img_right.header.stamp.to_sec(),
            'psm1_joint': psm1_joint.header.stamp.to_sec(),
            'psm1_jaw': psm1_jaw.header.stamp.to_sec(),
            'psm2_joint': psm2_joint.header.stamp.to_sec(),
            'psm2_jaw': psm2_jaw.header.stamp.to_sec(),
        }
        t2 = time.perf_counter()

        # Check timestamp sync
        max_ts = max(timestamps.values())
        min_ts = min(timestamps.values())
        sync_delta = (max_ts - min_ts) * 1000
        t3 = time.perf_counter()

        # Log callback rate
        # if self.prev_time is not None:
        #     callback_interval = (timestamps['img_left'] - self.prev_time) * 1000
        #     rospy.loginfo(f"Sync callback interval: {callback_interval:.1f}ms | Timestamp sync delta: {sync_delta:.1f}ms")
        self.prev_time = timestamps['img_left']
        t4 = time.perf_counter()

        # Check if recording
        if self.state != "RECORDING":
            return
        t5 = time.perf_counter()

        # Copy left image
        img_left_data = bytes(img_left.data)
        t6 = time.perf_counter()

        # Copy right image
        img_right_data = bytes(img_right.data)
        t7 = time.perf_counter()

        # Create numpy arrays one by one
        psm1_joint_arr = np.array(psm1_joint.position, dtype=np.float32)
        t8 = time.perf_counter()

        psm1_jaw_arr = np.array(psm1_jaw.position, dtype=np.float32)
        t9 = time.perf_counter()

        psm2_joint_arr = np.array(psm2_joint.position, dtype=np.float32)
        t10 = time.perf_counter()

        psm2_jaw_arr = np.array(psm2_jaw.position, dtype=np.float32)
        t11 = time.perf_counter()

        # Create dict
        frame_data = {
            'img_left_data': img_left_data,
            'img_left_shape': (img_left.height, img_left.width),
            'img_right_data': img_right_data,
            'img_right_shape': (img_right.height, img_right.width),
            'timestamp': timestamps['img_left'],
            'psm1_joint': psm1_joint_arr,
            'psm1_jaw': psm1_jaw_arr,
            'psm2_joint': psm2_joint_arr,
            'psm2_jaw': psm2_jaw_arr,
        }
        t12 = time.perf_counter()

        # Queue
        try:
            self.frame_queue.put_nowait(frame_data)
            t13 = time.perf_counter()

            # Check GC after callback
            # gc_count_after = gc.get_count()
            # gc_happened = (gc_count_after[0] < gc_count_before[0])  # Gen 0 count decreased = GC ran

            # Extensive timing breakdown - COMMENTED OUT: logging was causing 20-160ms delays!
            # overhead = (t1 - t0_enter) * 1000
            # ts_time = (t2 - t1) * 1000
            # sync_time = (t3 - t2) * 1000
            # log_time = (t4 - t3) * 1000
            # state_time = (t5 - t4) * 1000
            # left_time = (t6 - t5) * 1000
            # right_time = (t7 - t6) * 1000
            # psm1j_time = (t8 - t7) * 1000
            # psm1jw_time = (t9 - t8) * 1000
            # psm2j_time = (t10 - t9) * 1000
            # psm2jw_time = (t11 - t10) * 1000
            # dict_time = (t12 - t11) * 1000
            # queue_time = (t13 - t12) * 1000
            # total = (t13 - t0_enter) * 1000

            # gc_indicator = "🗑️ GC!" if gc_happened else ""
            # rospy.logwarn(f"⏱️  OVERHEAD: enter={overhead:.2f}ms ts={ts_time:.2f}ms sync={sync_time:.2f}ms log={log_time:.2f}ms state={state_time:.2f}ms {gc_indicator}")
            # rospy.logwarn(f"⏱️  IMAGES: left_copy={left_time:.2f}ms right_copy={right_time:.2f}ms")
            # rospy.logwarn(f"⏱️  JOINTS: psm1_j={psm1j_time:.2f}ms psm1_jw={psm1jw_time:.2f}ms psm2_jw={psm2jw_time:.2f}ms")
            # rospy.logwarn(f"⏱️  FINAL: dict={dict_time:.2f}ms queue={queue_time:.2f}ms | TOTAL={total:.2f}ms")
            # rospy.logwarn(f"🗑️  GC counts: before={gc_count_before} after={gc_count_after}")

            # queue_size = self.frame_queue.qsize()
            # if queue_size >= self.queue_warning_threshold:
            #     rospy.logwarn(f"⚠️  Encoding queue is {queue_size}% full - encoding may not be keeping up!")
        except queue.Full:
            rospy.logerr("❌ Frame queue FULL - DROPPING FRAME! Encoding cannot keep up with capture rate!")

    def _encoding_worker(self):
        """Background thread that encodes and writes frames to disk"""
        rospy.loginfo("Encoding thread started")

        while not rospy.is_shutdown():
            try:
                # Block until frame is available (with timeout for shutdown)
                frame_data = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Check if we should process (quick lock, then release)
            with self.state_lock:
                should_record = (self.state == "RECORDING" and
                               self.video_writer_left is not None)
                video_writer_left = self.video_writer_left
                video_writer_right = self.video_writer_right

            if should_record:
                try:
                    # Convert raw bytes to OpenCV BGR format (done in background)
                    img_left_np = np.frombuffer(frame_data['img_left_data'], dtype=np.uint8)
                    img_left_bgr = img_left_np.reshape(frame_data['img_left_shape'][0],
                                                       frame_data['img_left_shape'][1], 3)

                    img_right_np = np.frombuffer(frame_data['img_right_data'], dtype=np.uint8)
                    img_right_bgr = img_right_np.reshape(frame_data['img_right_shape'][0],
                                                         frame_data['img_right_shape'][1], 3)

                    # Encode WITHOUT holding lock (allows callback to run freely)
                    video_writer_left.write_frame(img_left_bgr)
                    video_writer_right.write_frame(img_right_bgr)

                    # Only lock when appending to buffer (very fast)
                    state_data = {
                        'timestamp': frame_data['timestamp'],
                        'psm1_joint': frame_data['psm1_joint'],
                        'psm1_jaw': frame_data['psm1_jaw'],
                        'psm2_joint': frame_data['psm2_joint'],
                        'psm2_jaw': frame_data['psm2_jaw'],
                    }

                    with self.state_lock:
                        # Double-check we're still recording
                        if self.state == "RECORDING":
                            self.episode_buffer.append(state_data)
                except Exception as e:
                    # Video writer was closed during encoding, ignore
                    rospy.logdebug(f"Frame dropped during episode stop: {e}")

            # Mark task as done for queue tracking
            self.frame_queue.task_done()
    def on_key_press(self, key):
        try:
            if key.char == 'b':
                with self.state_lock:
                    if self.state == "IDLE":
                        self.start_episode()
                    elif self.state == "RECORDING":
                        self.stop_episode()
        except AttributeError:
            pass

    def start_episode(self):
        """Called when 'b' is pressed to start recording"""
        # Note: Called while holding state_lock, but callbacks don't wait for lock

        self.episode_count += 1
        self.episode_buffer = []

        # Create episode directory
        episode_dir = f"{self.data_dir}/episode_{self.episode_count:04d}"
        os.makedirs(episode_dir, exist_ok=True)

        rospy.loginfo(f"Initializing encoders for episode {self.episode_count}...")

        # Initialize video writers (slow - GPU encoder initialization ~200-300ms)
        # We do this BEFORE setting state to RECORDING to avoid gaps
        self.video_writer_left = VideoWriter(
            f"{episode_dir}/camera_left.mp4",
            width=1280, height=960, fps=30
        )
        self.video_writer_right = VideoWriter(
            f"{episode_dir}/camera_right.mp4",
            width=1280, height=960, fps=30
        )

        # Only NOW set state to RECORDING - encoders are ready!
        self.state = "RECORDING"
        rospy.loginfo(f"✓ Started episode {self.episode_count}")
    
    def stop_episode(self):
        """Called when 'b' is pressed to stop recording"""
        self.state = "IDLE"

        # Wait for encoding queue to drain
        queue_size = self.frame_queue.qsize()
        if queue_size > 0:
            rospy.loginfo(f"Waiting for {queue_size} frames to finish encoding...")
            self.frame_queue.join()  # Block until all frames are processed
            rospy.loginfo("✓ All frames encoded")

        # EXPERIMENTAL: Re-enable GC and collect garbage after recording
        # Uncomment if you enabled gc.disable() in start_episode
        # gc.enable()
        # collected = gc.collect()
        # rospy.loginfo(f"🗑️  Re-enabled GC, collected {collected} objects")

        # Close video files (flushes encoder)
        self.video_writer_left.close()
        self.video_writer_right.close()

        # Save state data as NPZ
        episode_dir = f"{self.data_dir}/episode_{self.episode_count:04d}"

        timestamps = [f['timestamp'] for f in self.episode_buffer]
        psm1_joints = np.array([f['psm1_joint'] for f in self.episode_buffer])
        psm1_jaws = np.array([f['psm1_jaw'] for f in self.episode_buffer])
        psm2_joints = np.array([f['psm2_joint'] for f in self.episode_buffer])
        psm2_jaws = np.array([f['psm2_jaw'] for f in self.episode_buffer])

        np.savez_compressed(
            f"{episode_dir}/frames.npz",
            timestamps=timestamps,
            psm1_joint_pos=psm1_joints,
            psm1_jaw_pos=psm1_jaws,
            psm2_joint_pos=psm2_joints,
            psm2_jaw_pos=psm2_jaws,
        )

        # Save metadata
        metadata = {
            'episode_index': self.episode_count,
            'num_frames': len(self.episode_buffer),
            'duration_sec': timestamps[-1] - timestamps[0],
            'collection_date': datetime.now().isoformat(),
        }

        with open(f"{episode_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        rospy.loginfo(f"Saved episode {self.episode_count} "
                     f"({len(self.episode_buffer)} frames)")
        rospy.loginfo(f"FPS: {len(self.episode_buffer) / (timestamps[-1] - timestamps[0])}")

        self.episode_buffer = []

if __name__ == "__main__":
    # Initialize as ROS node
    rospy.init_node("data_collection_node")
    data_collection_node = DataCollectionNode()
    rospy.spin()