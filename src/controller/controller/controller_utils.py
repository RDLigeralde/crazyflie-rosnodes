import numpy as np
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry

def odom_to_body(msg: Odometry):
    p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    quat_wb = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    R_wb = R.from_quat(quat_wb).as_matrix()

    v_w = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
    v_b = R_wb.T @ v_w

    w_w = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
    w_b = R_wb.T @ w_w

    return p, v_b, w_b, R_wb, quat_wb