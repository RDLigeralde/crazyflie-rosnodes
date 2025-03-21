import os
import numpy as np
import time
from threading import Lock

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

from .controller_utils import odom_to_body

from scipy.spatial.transform import Rotation

from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj

def update_setpoint_clbk(self, request, response):
    if self.fsm.state == 'hovering':
        self.get_logger().info(f"Received new setpoint: {x0}")
    elif self.fsm.state == 'flying':
        self.fsm.hovering()
        self.get_logger().info(f"[FSM] Hovering at new setpoint: {x0}")
    else:
        self.get_logger().info("Cannot update setpoint from current state")
        response.success = False
        return response

    x0 = np.array([request.x, request.y, request.z])
    new_point = HoverTraj(x0=x0, yaw0=request.yaw).update(0)
    with self.traj_lock:
        self.flat_output = new_point

    response.success = True

    return response

def trajectory_clbk(self, _, response):
    if self.fsm.state != 'hovering':
        self.get_logger().info("Cannot start trajectory from current state")
        response.success = False
        return response

    self.logger.info(f"Starting circular trajectory")
    self.fsm.flying()

    with self.moca_lock:
        state = self.mocap_pose

    radius = 1.0
    center = np.array([state['x'][0], state['x'][1] + radius, state['x'][2]])
    freq = 0.1
    yaw_bool = False
    plane = 'XY'
    direction = 'CCW'

    circular_traj = CircularTraj(center=center, radius=radius, freq=freq, yaw_bool=yaw_bool, plane=plane, direction=direction)

    t0 = time.time()
    dt = 0
    while dt < 10.0:
        dt = time.time() - t0
        with self.traj_lock:
            self.flat_output = circular_traj.update(dt)
        time.sleep(0.1)

    self.logger.info(f"Finished circular trajectory")
    self.fsm.hovering()

    response.success = True

    return response

def takeoff_clbk(self, _, response):
    if self.mocap_pose == {}:
        self.get_logger().info("Mocap data not available yet")
        response.success = False
        return response

    if self.fsm.state != 'landed':
        self.get_logger().info("Cannot takeoff from current state")
        response.success = False
        return response

    t0 = time.time()

    # Change FSM state
    self.fsm.takeoff()
    self.get_logger().info("[FSM] Taking off")

    # Arm the Crazyflie
    self.scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)

    # Unlock startup thrust protection
    self.scf.cf.commander.send_setpoint(0, 0, 0, 0)

    with self.mocap_lock:
        state = self.mocap_pose

    x0 = [state['x'][0], state['x'][1], self.takeoff_height]
    takeoff_point = HoverTraj(x0=x0).update(0)
    with self.traj_lock:
        self.flat_output = takeoff_point

    while (time.time() - t0 < 3.0):
        time.sleep(0.2)

    # Change FSM state
    self.fsm.in_position()
    self.get_logger().info("[FSM] Hovering")

    response.success = True

    return response

def landing_clbk(self, _, response):
    print(self.fsm.state)
    if self.fsm.state == 'flying':
        self.get_logger().info("[FSM] Hovering")
        self.fsm.hovering()

        # Get the most recent mocap pose as hover point
        with self.mocap_lock:
            state = self.mocap_pose

        hover_point = HoverTraj(x0=state['x']).update(0)
        with self.traj:
            self.flat_output = hover_point

        time.sleep(1)

    elif self.fsm.state != 'hovering':
        self.get_logger().info("Cannot takeoff from current state")
        response.success = False
        return response

    self.get_logger().info("[FSM] Landing")
    self.fsm.land()

    with self.mocap_lock:
        state = self.mocap_pose

    x0 = [state['x'][0], state['x'][1], 0.05]
    landing_point = HoverTraj(x0=x0).update(0)
    with self.traj_lock:
        self.flat_output = landing_point

    t0 = time.time()
    while (time.time() - t0 < 3.0):
        time.sleep(0.2)

    for _ in range(30):
        self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.4)

    self.get_logger().info("[FSM] Landed")
    self.fsm.landing_complete()

    response.success = True

    return response

def cmd_clbk(self):
    if self.fsm.state not in ['hovering', 'flying']:
        return

    with self.mocap_lock:
        state = self.mocap_pose
    with self.traj_lock:
        traj = self.flat_output

    control = self.controller.update(0, state, traj)

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

def mocap_clbk(self, msg: Odometry):
    """
    Mocap odometry callback
    """
    p, v_b, w_b, R_wb, quat_wb = odom_to_body(msg)

    print("Position: ", p)
    # print("Velocity: ", v_b)
    # print("Angular velocity: ", w_b)
    # print("Rotation matrix: ", R_wb)
    # print("Quaternion: ", quat_wb)

    with self.mocap_lock:
        self.mocap_pose['x'] = p
        self.mocap_pose['v'] = v_b
        self.mocap_pose['w'] = w_b
        self.mocap_pose['R'] = R_wb
        self.mocap_pose['q'] = quat_wb
