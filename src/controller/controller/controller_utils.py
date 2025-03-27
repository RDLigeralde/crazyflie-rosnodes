import numpy as np
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from jirl_interfaces.msg import CommandCTBR, Trajectory

def send_ctbr_command(self, thrust_pwm, thrust_N, roll_rate, pitch_rate, yaw_rate):
    """
    Send CTBR command
    """
    command_msg = CommandCTBR()
    command_msg.crazyflie_name = self.get_namespace().split('/')[-1]
    command_msg.thrust_pwm = thrust_pwm
    command_msg.thrust_n = thrust_N
    command_msg.roll_rate = float(roll_rate)
    command_msg.pitch_rate = float(pitch_rate)
    command_msg.yaw_rate = float(yaw_rate)

    self.cmd_pub.publish(command_msg)

def send_trajectory(self, traj):
    """
    Send trajectory
    """
    traj_msg = Trajectory()
    traj_msg.x = traj['x']
    traj_msg.x_dot = traj['x_dot']
    traj_msg.x_ddot = traj['x_ddot']
    traj_msg.x_dddot = traj['x_dddot']
    traj_msg.x_ddddot = traj['x_ddddot']
    traj_msg.yaw = traj['yaw']
    traj_msg.yaw_dot = traj['yaw_dot']
    traj_msg.yaw_ddot = traj['yaw_ddot']

    self.traj_pub.publish(traj)