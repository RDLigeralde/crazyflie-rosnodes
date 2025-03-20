import os
import numpy as np
import time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

from .controller_utils import odom_to_body

from scipy.spatial.transform import Rotation

def update_setpoint_clbk(self, request, response):
    self.x_des = request.x
    self.y_des = request.y
    self.z_des = request.z
    self.yaw_des = request.yaw

    self.get_logger().info(f"Received new setpoint: ({self.x_des}, {self.y_des}, {self.z_des}, {self.yaw_des})")
    response.success = True

    return response

def takeoff_clbk(self, _, response):
    self.flat_output = self.trajectory.update(0)
    self.flat_output['x'] = [self.x_des, self.y_des, self.takeoff_height]

    self.state = 'flying'
    # Unlock startup thrust protection
    self.scf.cf.commander.send_setpoint(0, 0, 0, 0)

    response.success = True

    return response

def landing_clbk(self, _, response):
    self.t0 = time.time()

    self.flat_output = self.trajectory.update(0)
    self.flat_output['x'] = [self.x_des, self.y_des, 0.05]

    self.state = 'landing'
    response.success = True

    return response

def cmd_clbk(self):
    if len(self.mocap_pose) == 0:
        return

    if not hasattr(self, 'first_time'):
        self.first_time = False

        # Arm the Crazyflie
        self.scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        # Unlock startup thrust protection
        self.scf.cf.commander.send_setpoint(0, 0, 0, 0)

        self.x_des = self.mocap_pose['x'][0]
        self.y_des = self.mocap_pose['x'][1]
        self.z_des = self.takeoff_height

    if self.state == 'landing':
        if time.time() - self.t0 > 2.0:
            # self.scf.cf.commander.send_stop_setpoint()
            for _ in range(30):
                self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
                time.sleep(0.1)
            print("Landed")
            self.state = 'landed'
            return

    elif self.state == 'flying':
        print("p: ", self.mocap_pose['x'])
        print("v_b: ", self.mocap_pose['v'])
        print("w_b: ", self.mocap_pose['w'])
        print("q_wb: ", self.mocap_pose['q'])

        self.flat_output = self.trajectory.update(0)
        self.flat_output['x'] = [self.x_des, self.y_des, self.z_des]
    else:
        return

    control = self.controller.update(0, self.mocap_pose, self.flat_output)

    c1 = self.low_level_controller_c1
    c2 = self.low_level_controller_c2
    c3 = self.low_level_controller_c3
    thrust_pwm_min = self.low_level_controller_thrust_pwm_min
    thrust_pwm_max = self.low_level_controller_thrust_pwm_max

    thrust_des_newtons = control['cmd_thrust']
    print("Thrust des: ", thrust_des_newtons)
    thrust_des_grams = thrust_des_newtons / 9.81 * 1000
    print("Thrust des grams: ", thrust_des_grams)
    if c3 + thrust_des_grams < 0:
        print("Thrust too negative")
        thrust_des_grams = 0
    thrust_pwm = c1 + c2 * (c3 + thrust_des_grams)**.5
    print("Thrust PWM: ", thrust_pwm)
    thrust_pwm = min(max(0.0, thrust_pwm), 0.9)
    print("Thrust PWM: ", thrust_pwm)
    thrust_pwm = int(thrust_pwm * thrust_pwm_max)
    print("Thrust PWM: ", thrust_pwm)

    w_des = control['cmd_w']        # deg/s
    print("W des: ", w_des)
    print()

    self.scf.cf.commander.send_setpoint(w_des[0], w_des[1], -w_des[2], thrust_pwm)  # FIXME

    ########
    # # Desired TRPY
    # thrust_des_newtons = control['cmd_thrust']
    # thrust_des_grams = thrust_des_newtons / 9.81 * 1000
    # q_des = control['cmd_q']
    # R_des = Rotation.from_quat(q_des).as_matrix()
    # eul_des = Rotation.from_quat(q_des).as_euler('ZXY', degrees=True)
    # yaw_des = eul_des[0]

    # # Current RPY
    # R_cur = Rotation.from_quat(self.state['q'])
    # eul_cur = R_cur.as_euler('ZXY', degrees=True)
    # yaw_cur = eul_cur[0]

    # # Map the desired onto the current body frame based on yaw
    # R_z = Rotation.from_rotvec((yaw_cur - yaw_des)*(np.pi/180)*np.array([0,0,1])).as_matrix()
    # R_des_new = R_des@R_z

    # pitch_des = -np.arcsin(R_des_new[2,0])*180/np.pi
    # roll_des = np.arctan2(R_des_new[2,1], R_des_new[2,2])*180/np.pi

    # e_yaw = yaw_des - yaw_cur
    # if e_yaw > 180:
    #     e_yaw -= 360
    # elif e_yaw < -180:
    #     e_yaw += 360
    # yaw_rate_des = -kyaw * e_yaw

    # if c3 + thrust_des_grams < 0:
    #     print("Thrust too negative")
    #     thrust_des_grams = 0
    # thrust_pwm = c1 + c2 * (c3 + thrust_des_grams)**.5
    # thrust_pwm = int(min(max(0.0, thrust_pwm), 0.9) * thrust_pwm_max)

    # self.scf.cf.commander.send_setpoint(roll_des, pitch_des, yaw_rate_des, thrust_pwm)


def mocap_clbk(self, msg: Odometry):
    """
    Mocap odometry callback
    """

    # x = msg.pose.pose.position.x
    # y = msg.pose.pose.position.y
    # z = msg.pose.pose.position.z
    # self.position = np.array([x, y, z])
    # qx = msg.pose.pose.orientation.x
    # qy = msg.pose.pose.orientation.y
    # qz = msg.pose.pose.orientation.z
    # qw = msg.pose.pose.orientation.w
    # self.quat = np.array([qx, qy, qz, qw])
    # vx = msg.twist.twist.linear.x
    # vy = msg.twist.twist.linear.y
    # vz = msg.twist.twist.linear.z
    # self.velocity = np.array([vx, vy, vz])
    # wx = msg.twist.twist.angular.x
    # wy = msg.twist.twist.angular.y
    # wz = msg.twist.twist.angular.z
    # self.angular_velocity = np.array([wx, wy, wz])
    # print("Position: ", self.position)
    # print("Quaternion: ", self.quat)
    # print("Velocity: ", self.velocity)
    # print("Angular velocity: ", self.angular_velocity)
    # print()
    # parse msg
    p, v_b, w_b, R_wb, quat_wb = odom_to_body(msg)

    self.mocap_pose['x'] = p
    self.mocap_pose['v'] = v_b
    self.mocap_pose['w'] = w_b
    self.mocap_pose['R'] = R_wb
    self.mocap_pose['q'] = quat_wb
