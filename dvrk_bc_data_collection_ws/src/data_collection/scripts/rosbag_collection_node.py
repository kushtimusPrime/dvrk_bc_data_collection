#!/usr/bin/env python3
"""
Keyboard-controlled rosbag recording for dVRK data collection.
Press 'b' to start/stop episode recording.

This replaces the PyAV-based real-time encoding approach with simple
rosbag recording, eliminating the real-time encoding bottleneck.
"""

import rospy
from pynput import keyboard
import subprocess
import signal
import os
from datetime import datetime

class RosbagCollectionNode:
    def __init__(self):
        rospy.init_node('rosbag_collection_node')

        self.state = "IDLE"  # IDLE | RECORDING
        self.episode_count = 0
        self.rosbag_process = None
        self.data_dir = f"data/raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        rospy.loginfo(f"Data directory: {self.data_dir}")

        # Topics to record
        self.topics = [
            "/av/img_left_rect",
            "/av/img_right_rect",
            "/dvrk/PSM1/state_joint_current",
            "/dvrk/PSM1/state_jaw_current",
            "/dvrk/PSM2/state_joint_current",
            "/dvrk/PSM2/state_jaw_current",
        ]

        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

        rospy.loginfo("Rosbag collection node ready. Press 'b' to start/stop recording.")

    def on_key_press(self, key):
        try:
            if key.char == 'b':
                if self.state == "IDLE":
                    self.start_recording()
                elif self.state == "RECORDING":
                    self.stop_recording()
        except AttributeError:
            pass

    def start_recording(self):
        """Start rosbag recording for new episode"""
        self.episode_count += 1
        bag_path = f"{self.data_dir}/episode_{self.episode_count:04d}.bag"

        # Build rosbag record command with BZ2 compression
        cmd = ["rosbag", "record"] + self.topics + ["-O", bag_path, "-j"]

        # Launch rosbag as subprocess
        self.rosbag_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.state = "RECORDING"
        rospy.loginfo(f"✓ Started recording episode {self.episode_count}")

    def stop_recording(self):
        """Stop current rosbag recording"""
        if self.rosbag_process is None:
            return

        # Send SIGINT to rosbag (same as Ctrl+C)
        self.rosbag_process.send_signal(signal.SIGINT)

        rospy.loginfo(f"Stopping episode {self.episode_count}...")

        # Wait for rosbag to finish writing (timeout after 10s)
        try:
            self.rosbag_process.wait(timeout=10)
            rospy.loginfo(f"✓ Saved episode {self.episode_count}")
        except subprocess.TimeoutExpired:
            rospy.logerr("Rosbag did not stop cleanly - killing process")
            self.rosbag_process.kill()

        self.rosbag_process = None
        self.state = "IDLE"

    def shutdown(self):
        """Clean shutdown handler"""
        if self.state == "RECORDING":
            rospy.loginfo("Stopping active recording before shutdown...")
            self.stop_recording()
        
        rospy.loginfo("Shutting down rosbag collection node. Data directory: %s", self.data_dir)

if __name__ == "__main__":
    node = RosbagCollectionNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
