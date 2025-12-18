#!/usr/bin/env python3
"""
Test message synchronization for data collection.
Run for extended periods to find optimal slop parameter.

Usage:
    python test_synchronization.py --duration 300 --slop 0.033
"""

import rospy
import message_filters
from sensor_msgs.msg import Image, JointState
import numpy as np
import argparse
from collections import defaultdict

class SyncTester:
    def __init__(self, slop=0.033):
        rospy.init_node('sync_tester', anonymous=True)

        self.slop = slop
        self.sync_count = 0
        self.start_time = None
        self.last_sync_time = None

        # Track per-topic stats
        self.topic_stats = defaultdict(lambda: {'count': 0, 'timestamps': []})

        # Subscribe to all topics
        img_left_sub = message_filters.Subscriber("/av/img_left_rect", Image)
        img_right_sub = message_filters.Subscriber("/av/img_right_rect", Image)
        psm1_joint_sub = message_filters.Subscriber("/dvrk/PSM1/state_joint_current", JointState)
        psm1_jaw_sub = message_filters.Subscriber("/dvrk/PSM1/state_jaw_current", JointState)
        psm2_joint_sub = message_filters.Subscriber("/dvrk/PSM2/state_joint_current", JointState)
        psm2_jaw_sub = message_filters.Subscriber("/dvrk/PSM2/state_jaw_current", JointState)

        # Also subscribe individually to track total messages per topic
        rospy.Subscriber("/av/img_left_rect", Image,
                        lambda msg: self.topic_callback('img_left', msg))
        rospy.Subscriber("/av/img_right_rect", Image,
                        lambda msg: self.topic_callback('img_right', msg))
        rospy.Subscriber("/dvrk/PSM1/state_joint_current", JointState,
                        lambda msg: self.topic_callback('psm1_joint', msg))
        rospy.Subscriber("/dvrk/PSM1/state_jaw_current", JointState,
                        lambda msg: self.topic_callback('psm1_jaw', msg))
        rospy.Subscriber("/dvrk/PSM2/state_joint_current", JointState,
                        lambda msg: self.topic_callback('psm2_joint', msg))
        rospy.Subscriber("/dvrk/PSM2/state_jaw_current", JointState,
                        lambda msg: self.topic_callback('psm2_jaw', msg))

        # Synchronizer
        sync = message_filters.ApproximateTimeSynchronizer(
            [img_left_sub, img_right_sub, psm1_joint_sub,
             psm1_jaw_sub, psm2_joint_sub, psm2_jaw_sub],
            queue_size=10,
            slop=slop
        )
        sync.registerCallback(self.sync_callback)

    def topic_callback(self, topic_name, msg):
        """Track individual topic messages"""
        timestamp = msg.header.stamp.to_sec()
        self.topic_stats[topic_name]['count'] += 1
        self.topic_stats[topic_name]['timestamps'].append(timestamp)

    def sync_callback(self, img_left, img_right, psm1_joint, psm1_jaw, psm2_joint, psm2_jaw):
        """Called when all 6 messages are synchronized"""
        if self.start_time is None:
            self.start_time = rospy.Time.now()

        current_time = rospy.Time.now()
        self.sync_count += 1

        # Extract timestamps from all messages
        timestamps = [
            img_left.header.stamp.to_sec(),
            img_right.header.stamp.to_sec(),
            psm1_joint.header.stamp.to_sec(),
            psm1_jaw.header.stamp.to_sec(),
            psm2_joint.header.stamp.to_sec(),
            psm2_jaw.header.stamp.to_sec(),
        ]

        # Calculate sync metrics
        ts_array = np.array(timestamps)
        max_diff = (ts_array.max() - ts_array.min()) * 1000  # ms

        # Log every 30 synced frames
        if self.sync_count % 30 == 0:
            elapsed = (current_time - self.start_time).to_sec()
            sync_rate = self.sync_count / elapsed if elapsed > 0 else 0

            rospy.loginfo(f"Synced: {self.sync_count} | Rate: {sync_rate:.1f} Hz | "
                         f"Max timestamp diff: {max_diff:.1f} ms")

        self.last_sync_time = current_time

    def print_summary(self):
        """Print final statistics"""
        print("\n" + "="*70)
        print("SYNCHRONIZATION TEST SUMMARY")
        print("="*70)

        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        sync_rate = self.sync_count / elapsed if elapsed > 0 else 0

        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Slop parameter: {self.slop*1000:.1f} ms")
        print(f"Synchronized frames: {self.sync_count}")
        print(f"Average sync rate: {sync_rate:.1f} Hz")
        print()

        print("Per-Topic Message Counts:")
        print("-" * 70)

        # Calculate drop rate
        min_count = min([stats['count'] for stats in self.topic_stats.values()])

        for topic_name, stats in sorted(self.topic_stats.items()):
            count = stats['count']
            expected_rate = count / elapsed if elapsed > 0 else 0
            drop_rate = ((count - self.sync_count) / count * 100) if count > 0 else 0

            print(f"{topic_name:20s}: {count:5d} msgs | {expected_rate:5.1f} Hz | "
                  f"{drop_rate:5.1f}% not synced")

        print()
        overall_drop = ((min_count - self.sync_count) / min_count * 100) if min_count > 0 else 0
        print(f"Overall drop rate: {overall_drop:.2f}% (messages published but not synced)")
        print()

        if overall_drop < 1.0:
            print("✓ EXCELLENT: <1% drop rate - synchronization working well!")
        elif overall_drop < 5.0:
            print("✓ GOOD: <5% drop rate - acceptable for data collection")
        elif overall_drop < 10.0:
            print("⚠ WARNING: 5-10% drop rate - consider increasing slop or investigating delays")
        else:
            print("✗ POOR: >10% drop rate - increase slop or check for system issues")

        print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Test ROS message synchronization')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--slop', type=float, default=0.033,
                       help='Slop parameter in seconds (default: 0.033 = 33ms)')

    args = parser.parse_args()

    print(f"Starting synchronization test for {args.duration} seconds with slop={args.slop*1000:.1f}ms")
    print("Press Ctrl+C to stop early and see results")
    print()

    tester = SyncTester(slop=args.slop)

    try:
        rospy.sleep(args.duration)
    except rospy.ROSInterruptException:
        pass

    tester.print_summary()

if __name__ == '__main__':
    main()