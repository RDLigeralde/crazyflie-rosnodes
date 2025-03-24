import os
import numpy as np
import time
from threading import Lock

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

from .controller_utils import odom_to_body

from jirl_interfaces.srv import UpdateSetpoint, Trajectory

from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj

def update_setpoint_clbk(self, request, response):
    x0 = np.array([request.x, request.y, request.z])

    if self.fsm.state == 'hovering':
        self.get_logger().info(f"Received new setpoint: {x0}")
    elif self.fsm.state == 'flying':
        self.fsm.hovering()
        self.get_logger().info(f"[FSM] Hovering at new setpoint: {x0}")
    else:
        self.get_logger().info("Cannot update setpoint from current state")
        response.success = False
        return response

    with self.traj_lock:
        self.flat_output = HoverTraj(x0=x0, yaw0=request.yaw).update(0)

    response.success = True
    return response

def trajectory_clbk(self, request, response):
    if self.fsm.state != 'hovering':
        self.get_logger().info("Cannot start trajectory from current state")
        response.success = False
        return response

    state = self.mocap_pose
    self.t0 = time.time()

    if request.trajectory_type == Trajectory.CIRCLE:
        center = np.array([state['x'][0] - radius, state['x'][1], state['x'][2]])
        radius = request.radius
        freq = request.freq
        yaw_bool = request.direction
        if request.plane == Trajectory.PLANE_XY:
            plane = 'XY'
        elif request.plane == Trajectory.PLANE_YZ:
            plane = 'YZ'
        elif request.plane == Trajectory.PLANE_XZ:
            plane = 'XZ'
        direction = 'CW' if request.direction == Trajectory.DIR_CW else 'CCW'
        self.traj_duration = request.duration

        self.trajectory = CircularTraj(center=center, radius=radius, freq=freq, yaw_bool=yaw_bool, plane=plane, direction=direction)

        self.get_logger().info(f"Starting circular trajectory")

    self.fsm.move()

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

    self.t0 = time.time()
    self.dt = 0.0

    # Unlock startup thrust protection
    self.scf.cf.commander.send_setpoint(0, 0, 0, 0)

    # Change FSM state
    self.fsm.takeoff()
    self.get_logger().info("[FSM] Taking off")

    response.success = True
    return response

def landing_clbk(self, _, response):
    if self.fsm.state != 'hovering':
        self.get_logger().info("Cannot land from current state")
        response.success = False
        return response

    self.t0 = time.time()

    self.get_logger().info("[FSM] Landing")
    self.fsm.land()

    response.success = True
    return response

def logger_clbk(self):
    for log_entry in self.sync_logger.next():
        print(log_entry)
        timestamp = log_entry[0]
        data = log_entry[1]
        name = log_entry[2]

        self.get_logger().info('[%d][%s]: %.3s' % (timestamp, name, data))

def mocap_clbk(self, msg: Odometry):
    """
    Mocap odometry callback
    """
    p, v_b, w_b, R_wb, quat_wb = odom_to_body(msg)

    # print("Position: ", p)
    # print("Velocity: ", v_b)
    # print("Angular velocity: ", w_b)
    # print("Rotation matrix: ", R_wb)
    # print("Quaternion: ", quat_wb)

    self.mocap_pose['x'] = p
    self.mocap_pose['v'] = v_b
    self.mocap_pose['w'] = w_b
    self.mocap_pose['R'] = R_wb
    self.mocap_pose['q'] = quat_wb

    if self.fsm.state == 'landed':
        return
    elif self.fsm.state == 'taking_off':
        # # Arm the Crazyflie
        # self.scf.cf.platform.send_arming_request(True)
        # time.sleep(1.0)

        x0 = [p[0], p[1], self.takeoff_height]
        self.flat_output = HoverTraj(x0=x0).update(0)

        if (time.time() - self.t0 > 3.0):
            # Change FSM state
            self.fsm.in_position()
            self.get_logger().info("[FSM] Hovering")
    elif self.fsm.state == 'hovering':
        pass
    elif self.fsm.state == 'landing':
        x0 = [p[0], p[1], 0.10]
        self.flat_output = HoverTraj(x0=x0).update(0)

        if (time.time() - self.t0 > 3.0):
            for _ in range(30):
                self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
                time.sleep(0.1)

            self.get_logger().info("[FSM] Landed")
            self.fsm.landing_complete()

            return
    elif self.fsm.state == 'flying':
        if self.dt > self.traj_duration:
            self.get_logger().info(f"Finished circular trajectory")
            self.fsm.stop()

        self.dt = time.time() - self.t0
        self.flat_output = self.trajectory.update(self.dt)

    # Apply control
    control = self.controller.update(0, self.mocap_pose, self.flat_output)

    c1 = self.low_level_controller_c1
    c2 = self.low_level_controller_c2
    c3 = self.low_level_controller_c3
    thrust_pwm_min = self.low_level_controller_thrust_pwm_min
    thrust_pwm_max = self.low_level_controller_thrust_pwm_max

    thrust_des_newtons = control['cmd_thrust']
    thrust_des_grams = thrust_des_newtons / 9.81 * 1000
    if c3 + thrust_des_grams < 0:
        # print("Thrust too negative")
        thrust_des_grams = 0
    thrust_pwm = c1 + c2 * (c3 + thrust_des_grams)**.5
    thrust_pwm = thrust_pwm * thrust_pwm_max + thrust_pwm_min * 1.0
    thrust_pwm = int(min(max(thrust_pwm_min, thrust_pwm), thrust_pwm_max * 0.9))

    w_des = control['cmd_w']        # deg/s

    self.scf.cf.commander.send_setpoint(w_des[0], w_des[1], -w_des[2], thrust_pwm)  # FIXME