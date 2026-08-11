import numpy as np
from scipy.spatial.transform import Rotation as R

def init_parameters(self):
    """
    Init parameters
    """
    # Declare parameters
    self.declare_parameters(
        namespace='',
        parameters=[('crazyradio_driver.enable', True),
                    ('crazyradio_driver.crazyflie_names', ['']),
                    ('crazyradio_driver.crazyradio_uris', ['']),
                    ('crazyradio_driver.ext_driver', ''),
                    ('gate_side', 0.0),
                    ('low_level_controller.c1', 0.0),
                    ('low_level_controller.c2', 0.0),
                    ('low_level_controller.c3', 0.0),
                    ('low_level_controller.thrust_pwm_min', 0),
                    ('low_level_controller.thrust_pwm_max', 0),
                    ('policy.path', ''),
                    ('policy.jax_enable', False),
                    ('policy.yaw_kp', 1.0),
                    ('policy.waypoints', [0.0]),
                    ('policy.initial_waypoint', 0),
                    ('policy.max_roll_br', 0.0),
                    ('policy.max_pitch_br', 0.0),
                    ('policy.max_yaw_br', 0.0),
                    ('policy.pass_gate_thr', 0.0),
                    ('takeoff_height', 0.0),
                    ('use_cond', True),
                   ])

    # Get parameters
    self.driver_enable = self.get_parameter('crazyradio_driver.enable').value
    self.driver_names = self.get_parameter('crazyradio_driver.crazyflie_names').value
    self.driver_uris = self.get_parameter('crazyradio_driver.crazyradio_uris').value
    self.driver_ext = self.get_parameter('crazyradio_driver.ext_driver').value
    self.gate_side = self.get_parameter('gate_side').value
    self.low_level_controller_c1 = self.get_parameter('low_level_controller.c1').value
    self.low_level_controller_c2 = self.get_parameter('low_level_controller.c2').value
    self.low_level_controller_c3 = self.get_parameter('low_level_controller.c3').value
    self.low_level_controller_thrust_pwm_min = self.get_parameter('low_level_controller.thrust_pwm_min').value
    self.low_level_controller_thrust_pwm_max = self.get_parameter('low_level_controller.thrust_pwm_max').value
    self.policy_path = self.get_parameter('policy.path').value
    # jax_enable selects JaxRacingPolicy (mjc_dronetests gate-racing
    # checkpoints, config.json + params.pkl at policy_path) over the
    # legacy PyTorch waypoint-tracking RacingPolicy — see
    # controller_node.py's init_controllers(). yaw_kp is only read by
    # JaxRacingPolicy's action_type="mellinger" path (its on-host
    # desired-yaw -> yaw-rate P-wrapper gain, see its own docstring) and
    # otherwise unused.
    self.policy_jax_enable = self.get_parameter('policy.jax_enable').value
    self.policy_yaw_kp = self.get_parameter('policy.yaw_kp').value
    self.policy_max_roll_br = self.get_parameter('policy.max_roll_br').value
    self.policy_max_pitch_br = self.get_parameter('policy.max_pitch_br').value
    self.policy_max_yaw_br = self.get_parameter('policy.max_yaw_br').value
    self.policy_pass_gate_thr = self.get_parameter('policy.pass_gate_thr').value
    self.takeoff_height = self.get_parameter('takeoff_height').value
    waypoints_flat = np.array(self.get_parameter('policy.waypoints').value, dtype=np.float32)
    self.waypoints = waypoints_flat.reshape(-1, 6)
    self.initial_waypoint = self.get_parameter('policy.initial_waypoint').value
    self.use_cond = self.get_parameter('use_cond').value

    # Print parameters
    self.get_logger().info(f'crazyradio_enable: {self.driver_enable}')
    self.get_logger().info(f'crazyradio_uris: {self.driver_names}')
    self.get_logger().info(f'crazyradio_driver: {self.driver_uris}')
    self.get_logger().info(f'crazyradio_external_driver: {self.driver_ext}')

    self.get_logger().info(f'c1: {self.low_level_controller_c1}')
    self.get_logger().info(f'c2: {self.low_level_controller_c2}')
    self.get_logger().info(f'c3: {self.low_level_controller_c3}')
    self.get_logger().info(f'gate_side: {self.gate_side}')
    self.get_logger().info(f'thrust_pwm_min: {self.low_level_controller_thrust_pwm_min}')
    self.get_logger().info(f'thrust_pwm_max: {self.low_level_controller_thrust_pwm_max}')
    self.get_logger().info(f'policy_path: {self.policy_path}')
    self.get_logger().info(f'policy_waypoints:\n{self.waypoints}')
    self.get_logger().info(f'policy_max_roll_br: {self.policy_max_roll_br}')
    self.get_logger().info(f'policy_max_pitch_br: {self.policy_max_pitch_br}')
    self.get_logger().info(f'policy_max_yaw_br: {self.policy_max_yaw_br}')
    self.get_logger().info(f'policy_pass_gate_thr: {self.policy_pass_gate_thr}')
    self.get_logger().info(f'initial_waypoint: {self.initial_waypoint}')
    self.get_logger().info(f'takeoff_height: {self.takeoff_height}')
    self.get_logger().info(f'use_cond: {self.use_cond}')
    self.get_logger().info(f'policy_jax_enable: {self.policy_jax_enable}')
    self.get_logger().info(f'policy_yaw_kp: {self.policy_yaw_kp}')

    #
    self.waypoints_quat = np.zeros((self.waypoints.shape[0], 4), dtype=np.float32)
    self.params = {
        "waypoints": self.waypoints,
        "waypoints_quat": self.waypoints_quat,
        "gate_side": self.gate_side,
        "initial_waypoint": self.initial_waypoint,
        "max_roll_br": self.policy_max_roll_br,
        "max_pitch_br": self.policy_max_pitch_br,
        "max_yaw_br": self.policy_max_yaw_br,
        "pass_gate_thr": self.policy_pass_gate_thr,
        # JaxRacingPolicy's own param names (action_type="attitude"'s
        # fallback-only rate scale, and action_type="mellinger"'s yaw
        # P-wrapper gain) — reuses the same policy.max_*_br values rather
        # than adding duplicate params, since both name the same real
        # calibration constant (a body-rate scale in deg/s).
        "max_roll_rate_dps": self.policy_max_roll_br,
        "max_pitch_rate_dps": self.policy_max_pitch_br,
        "max_yaw_rate_dps": self.policy_max_yaw_br,
        "yaw_kp": self.policy_yaw_kp,
    }

    for i, waypoint_data in enumerate(self.waypoints):
        euler_np = waypoint_data[3:6]
        rot_from_euler = R.from_euler('xyz', euler_np)
        self.waypoints_quat[i, :] = rot_from_euler.as_quat(scalar_first=True)
