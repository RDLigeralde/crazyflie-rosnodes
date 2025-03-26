import numpy as np
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from jirl_interfaces.msg import CommandCTBR

def odom_to_body(msg: Odometry):
    p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    quat_wb = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    R_wb = R.from_quat(quat_wb).as_matrix()

    v_w = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
    v_b = R_wb.T @ v_w

    w_w = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
    w_b = R_wb.T @ w_w

    return p, v_b, w_b, R_wb, quat_wb

def send_ctbr_command(self, thrust_pwm, roll_rate, pitch_rate, yaw_rate):
    """
    Send CTBR command
    """
    command = CommandCTBR()
    command.crazyflie_name = self.get_namespace().split('/')[-1]
    command.roll_rate = float(roll_rate)
    command.pitch_rate = float(pitch_rate)
    command.yaw_rate = float(yaw_rate)
    command.thrust_pwm = thrust_pwm

    self.cmd_pub.publish(command)