#!/usr/bin/env python

import sys
import rclpy
from rclpy.executors import SingleThreadedExecutor
from capture_the_flag.ctf_node import CaptureTheFlagNode

def main():
    # Initialize ROS 2 context and node
    rclpy.init(args=sys.argv)
    capture_the_flag_node = CaptureTheFlagNode()

    executor = SingleThreadedExecutor()
    executor.add_node(capture_the_flag_node)

    # Run the node
    try:
        executor.spin()
    except KeyboardInterrupt:
        capture_the_flag_node.cleanup()
        capture_the_flag_node.destroy_node()
        executor.shutdown()

if __name__ == '__main__':
    main()
