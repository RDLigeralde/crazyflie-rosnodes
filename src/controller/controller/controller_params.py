import numpy as np
from scipy.spatial.transform import Rotation as R

def init_parameters(self):
    """
    Init parameters
    """
    # Declare parameters
    self.declare_parameters(
        namespace='',
        parameters=[('crazyradio_driver', ''),
                    ('gate_side', 0.0),
                    ('low_level_controller.c1', 0.0),
                    ('low_level_controller.c2', 0.0),
                    ('low_level_controller.c3', 0.0),
                    ('low_level_controller.thrust_pwm_min', 0),
                    ('low_level_controller.thrust_pwm_max', 0),
                    ('policy.path', ''),
                    ('policy.waypoints', [0.0]),
                    ('takeoff_height', 0.5),
                   ])

    # Get parameters
    self.crazyradio_driver = self.get_parameter('crazyradio_driver').value
    self.gate_side = self.get_parameter('gate_side').value
    self.low_level_controller_c1 = self.get_parameter('low_level_controller.c1').value
    self.low_level_controller_c2 = self.get_parameter('low_level_controller.c2').value
    self.low_level_controller_c3 = self.get_parameter('low_level_controller.c3').value
    self.low_level_controller_thrust_pwm_min = self.get_parameter('low_level_controller.thrust_pwm_min').value
    self.low_level_controller_thrust_pwm_max = self.get_parameter('low_level_controller.thrust_pwm_max').value
    self.policy_path = self.get_parameter('policy.path').value
    self.takeoff_height = self.get_parameter('takeoff_height').value
    waypoints_flat = np.array(self.get_parameter('policy.waypoints').value, dtype=np.float32)
    self.waypoints = waypoints_flat.reshape(-1, 6)

    # Print parameters
    self.get_logger().info(f'crazyradio_driver: {self.crazyradio_driver}')
    self.get_logger().info(f'c1: {self.low_level_controller_c1}')
    self.get_logger().info(f'c2: {self.low_level_controller_c2}')
    self.get_logger().info(f'c3: {self.low_level_controller_c3}')
    self.get_logger().info(f'gate_side: {self.gate_side}')
    self.get_logger().info(f'thrust_pwm_min: {self.low_level_controller_thrust_pwm_min}')
    self.get_logger().info(f'thrust_pwm_max: {self.low_level_controller_thrust_pwm_max}')
    self.get_logger().info(f'policy_path: {self.policy_path}')
    self.get_logger().info(f'policy_waypoints:\n{self.waypoints}')
    self.get_logger().info(f'takeoff_height: {self.takeoff_height}')

    #
    self.waypoints_quat = np.zeros((self.waypoints.shape[0], 4), dtype=np.float32)

    for i, waypoint_data in enumerate(self.waypoints):
        euler_np = waypoint_data[3:6]
        rot_from_euler = R.from_euler('xyz', euler_np)
        self.waypoints_quat[i, :] = rot_from_euler.as_quat(scalar_first=True)
