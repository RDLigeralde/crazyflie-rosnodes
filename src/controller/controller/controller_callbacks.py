import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Odometry
from jirl_interfaces.srv import StartTrajectory

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
    self.dt = 0.0

    if request.trajectory_type == StartTrajectory.Request.CIRCLE:
        radius = request.radius
        center = np.array([state['x'][0] - radius, state['x'][1], state['x'][2]])
        freq = request.freq
        yaw_bool = request.direction
        if request.plane == StartTrajectory.Request.PLANE_XY:
            plane = 'XY'
        elif request.plane == StartTrajectory.Request.PLANE_YZ:
            plane = 'YZ'
        elif request.plane == StartTrajectory.Request.PLANE_XZ:
            plane = 'XZ'
        direction = 'CW' if request.direction == StartTrajectory.Request.DIR_CW else 'CCW'
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
    self.p0 = self.mocap_pose['x']

    # Unlock startup thrust protection
    self.send_ctbr_command(0, 0.0, 0.0, 0.0)

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
    self.p0 = self.mocap_pose['x']

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
                self.send_ctbr_command(0, 0.0, 0.0, 0.0)
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

    # Publish trajectory
    self.send_trajectory(self.flat_output)

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
    thrust_pwm = int(min(max(thrust_pwm_min, thrust_pwm), thrust_pwm_max * 0.95))

    w_des = control['cmd_w']        # deg/s

    # Publish command msg
    self.send_ctbr_command(thrust_pwm, thrust_des_newtons, w_des[0], w_des[1], w_des[2])