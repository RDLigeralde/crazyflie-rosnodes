import time
import numpy as np
from jirl_interfaces.msg import CommandCTBR, CommandAction, CommandAttitude, Trajectory, Observations
from nav_msgs.msg import Odometry

from rotorpy.trajectories.hover_traj import HoverTraj

from scipy.spatial.transform import Rotation as R

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

def send_action_command(self, action):
    """Custom-controller / open-loop-mixer dispatch (action_type="attitude"
    JaxRacingPolicy checkpoints): stream the policy's raw 4-float action
    [thrust_norm, roll, pitch, yaw_rate] for crazyflie-firmware's
    examples/app_race_policy (race_controller.c/mixer.c) to consume
    on-device, bypassing PID entirely — see Crazyflie::sendRaceAction (the
    receiving end: action_channel.c). seq/tx_tick_ms let the firmware
    reject stale/reordered packets and measure the true send period, same
    fields action_channel.c's ActionPacket expects.
    """
    self._action_seq = (getattr(self, '_action_seq', 0) + 1) & 0xFF
    command_msg = CommandAction()
    command_msg.crazyflie_name = self.get_namespace().split('/')[-1]
    command_msg.seq = self._action_seq
    command_msg.tx_tick_ms = int(time.time() * 1000) & 0xFFFF
    command_msg.action = [float(a) for a in action]

    self.action_pub.publish(command_msg)

def send_attitude_command(self, roll_deg, pitch_deg, yaw_rate_dps, thrust_pwm):
    """Onboard-Mellinger dispatch (action_type="mellinger" JaxRacingPolicy
    checkpoints): a real attitude setpoint (roll/pitch as angles, yaw
    closed into a rate by JaxRacingPolicy's own P-wrapper — see
    controller_policy_jax.py's update()) for firmware's stock Mellinger
    controller (stabilizer.controller=2, set at connect time — see
    crazyradio_driver's connect sequence) to consume via the legacy RPYT
    setpoint. Deliberately a separate message/topic from CommandCTBR/
    CommandAction — its roll/pitch fields are angles, not rates, and
    reusing CommandCTBR's rate-named fields for angle values would be a
    silent unit mismatch waiting to happen.
    """
    command_msg = CommandAttitude()
    command_msg.crazyflie_name = self.get_namespace().split('/')[-1]
    command_msg.roll_deg = float(roll_deg)
    command_msg.pitch_deg = float(pitch_deg)
    command_msg.yaw_rate_dps = float(yaw_rate_dps)
    command_msg.thrust_pwm = int(thrust_pwm)

    self.attitude_pub.publish(command_msg)

def send_trajectory(self, traj):
    """
    Send trajectory
    """
    traj_msg = Trajectory()
    traj_msg.x = np.array(traj['x'], dtype=np.float64)
    traj_msg.x_dot = np.array(traj['x_dot'], dtype=np.float64)
    traj_msg.x_ddot = np.array(traj['x_ddot'], dtype=np.float64)
    traj_msg.x_dddot = np.array(traj['x_dddot'], dtype=np.float64)
    traj_msg.x_ddddot = np.array(traj['x_ddddot'], dtype=np.float64)
    traj_msg.yaw = float(traj['yaw'])
    traj_msg.yaw_dot = float(traj['yaw_dot'])
    traj_msg.yaw_ddot = float(traj['yaw_ddot'])

    self.traj_pub.publish(traj_msg)

def single_update(self, msg: Odometry):
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
    self.mocap_pose['v_b'] = v_b
    self.mocap_pose['w_b'] = w_b
    self.mocap_pose['v_w'] = v_w
    self.mocap_pose['w_w'] = w_w

    thrust_pwm_min = self.low_level_controller_thrust_pwm_min
    thrust_pwm_max = self.low_level_controller_thrust_pwm_max

    if self.fsm.state == 'racing':
        control, obs = self.policy.update(self.mocap_pose)

        # Observations.msg's layout is RacingPolicy's own (waypoint/corners-
        # based, 38-dim) — JaxRacingPolicy's v3 obs is a different 21-dim
        # layout, so only publish this telemetry for the policy it actually
        # describes rather than silently mis-slicing a differently-shaped
        # array.
        action_type = getattr(self.policy, 'action_type', None)
        if action_type is None:
            obs_msg = Observations()
            obs_msg.lin_vel = obs[0:3]
            obs_msg.rot = obs[3:12]
            obs_msg.corners_pos_b_curr = obs[12:24]
            obs_msg.corners_pos_b_next = obs[24:36]
            obs_msg.cond = obs[36:38]

            self.obs_pub.publish(obs_msg)

        if action_type in ('attitude', 'mellinger'):
            # Both new paths dispatch by publishing to a topic a separate
            # crazyradio_driver node consumes (see send_action_command/
            # send_attitude_command) — that's only reachable when
            # driver_enable=False (the documented default: one controller
            # node per drone namespace, a separate driver node per drone).
            # driver_enable=True (embedded driver, one controller node
            # flying several drones via multi_mocap_clbk's direct
            # scf.cf.commander.send_setpoint call) has no equivalent direct
            # dispatch for these two paths yet, and that mode already has
            # known multi-drone state-sharing problems independent of this
            # change — fail loud rather than silently publish to a topic
            # nothing consumes.
            if self.driver_enable:
                self.get_logger().error(
                    f"action_type={action_type!r} is not supported with "
                    f"crazyradio_driver.enable=True (embedded driver) — use "
                    f"a separate crazyradio_driver node instead."
                )
                return
            if action_type == 'attitude':
                # Custom controller / open-loop mixer path — see
                # send_action_command's docstring. Bypasses
                # send_ctbr_command entirely; firmware's on-device mixer
                # computes thrust, not this workstation.
                self.send_action_command(control['raw_action'])
            else:
                # Onboard-Mellinger path — see send_attitude_command's
                # docstring. Also bypasses send_ctbr_command: this is a
                # real angle setpoint, not a CTBR rate setpoint.
                roll_deg, pitch_deg, yaw_rate_dps = control['cmd_attitude']
                thrust_pwm = int(thrust_pwm_min + control['cmd_thrust'] * (thrust_pwm_max - thrust_pwm_min))
                self.send_attitude_command(roll_deg, pitch_deg, yaw_rate_dps, thrust_pwm)
            return

        # action_type is None: the legacy PyTorch RacingPolicy, unaffected
        # by the attitude/mellinger split above — same CTBR path every
        # other FSM state below also uses.
        thrust_des_perc = control['cmd_thrust']
        thrust_pwm = int(thrust_pwm_min + thrust_des_perc * (thrust_pwm_max - thrust_pwm_min))

        thrust_des_newtons = thrust_des_perc * (0.038 * 9.81 * 3.15)
    else:
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

                return 0, 0.0, 0.0, 0.0
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
        control = self.se3_controller.update(0, self.mocap_pose, self.flat_output)

        c1 = self.low_level_controller_c1
        c2 = self.low_level_controller_c2
        c3 = self.low_level_controller_c3

        thrust_des_newtons = control['cmd_thrust']
        thrust_des_grams = thrust_des_newtons / 9.81 * 1000
        if c3 + thrust_des_grams < 0:
            thrust_des_grams = 0
        thrust_pwm = c1 + c2 * (c3 + thrust_des_grams)**.5
        thrust_pwm = thrust_pwm * thrust_pwm_max + thrust_pwm_min * 1.0
        thrust_pwm = int(min(max(thrust_pwm_min, thrust_pwm), thrust_pwm_max))

    w_des = control['cmd_w']        # deg/s

    # Publish command msg
    self.send_ctbr_command(thrust_pwm, thrust_des_newtons, w_des[0], w_des[1], w_des[2])

    return thrust_pwm, w_des[0], w_des[1], w_des[2]
