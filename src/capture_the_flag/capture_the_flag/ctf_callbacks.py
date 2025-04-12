import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import OdometryArray

def mocap_clbk(self, msg: OdometryArray):
    """
    Mocap odometry callback
    """
    for i in range(len(msg.odom_array)):









    p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    v_w = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
    w_w = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
    quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    R_mat = R.from_quat(quat).as_matrix()
    v_b = R_mat.T @ v_w
    w_b = R_mat.T @ w_w

    self.mocap_pose['x'] = p
    self.mocap_pose['R'] = R_mat
    self.mocap_pose['q'] = quat
    self.mocap_pose['yaw'] = R.from_matrix(R_mat).as_euler('zyx')[0]
    if self.policy_enabled:
        self.mocap_pose['v'] = v_b
        self.mocap_pose['w'] = w_b
    else:
        self.mocap_pose['v'] = v_w
        self.mocap_pose['w'] = w_w

    if self.fsm.state == 'landed':
        return
    elif self.fsm.state == 'taking_off':
        x0 = [self.p0[0], self.p0[1], self.takeoff_height]
        self.flat_output = HoverTraj(x0=x0).update(0)

        if (time.time() - self.t0 > 3.0):
            # Change FSM state
            self.fsm.in_position()
            self.get_logger().info("[FSM] Hovering")
    elif self.fsm.state == 'hovering':
        pass
    elif self.fsm.state == 'landing':
        if (time.time() - self.t0 < 1.0):
            x0 = [self.p0[0], self.p0[1], 0.15]
        else:
            x0 = [self.p0[0], self.p0[1], 0.05]
        self.flat_output = HoverTraj(x0=x0).update(0)

        if (time.time() - self.t0 > 3.0):
            for _ in range(30):
                self.send_ctbr_command(0, 0.0, 0.0, 0.0, 0.0)
                time.sleep(0.1)

            self.get_logger().info("[FSM] Landed")
            self.fsm.landing_complete()

            return
    elif self.fsm.state == 'flying':
        if self.dt > self.traj_duration:
            self.get_logger().info(f"Finished circular trajectory")
            self.flat_output = HoverTraj(x0=p).update(0)
            self.fsm.stop()
        else:
            self.dt = time.time() - self.t0
            self.flat_output = self.trajectory.update(self.dt)

    # Publish trajectory
    self.send_trajectory(self.flat_output)

    # Apply control
    control = self.capture_the_flag.update(0, self.mocap_pose, self.flat_output)

    c1 = self.low_level_capture_the_flag_c1
    c2 = self.low_level_capture_the_flag_c2
    c3 = self.low_level_capture_the_flag_c3
    thrust_pwm_min = self.low_level_capture_the_flag_thrust_pwm_min
    thrust_pwm_max = self.low_level_capture_the_flag_thrust_pwm_max

    thrust_des_newtons = control['cmd_thrust']
    thrust_des_grams = thrust_des_newtons / 9.81 * 1000
    if c3 + thrust_des_grams < 0:
        # print("Thrust too negative")
        thrust_des_grams = 0
    thrust_pwm = c1 + c2 * (c3 + thrust_des_grams)**.5
    thrust_pwm = thrust_pwm * thrust_pwm_max + thrust_pwm_min * 1.0
    thrust_pwm = int(min(max(thrust_pwm_min, thrust_pwm), thrust_pwm_max * 0.95))

    w_des = control['cmd_w']        # deg/s

    # Publish command msg
    self.send_ctbr_command(thrust_pwm, thrust_des_newtons, w_des[0], w_des[1], w_des[2])