import torch
from scipy.spatial.transform import Rotation as R

def init_parameters(self):
    """
    Init parameters
    """
    # Declare parameters
    self.declare_parameters(
        namespace='',
        parameters=[('gate_side', 0.0),
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
    self.gate_side = self.get_parameter('gate_side').value
    self.low_level_controller_c1 = self.get_parameter('low_level_controller.c1').value
    self.low_level_controller_c2 = self.get_parameter('low_level_controller.c2').value
    self.low_level_controller_c3 = self.get_parameter('low_level_controller.c3').value
    self.low_level_controller_thrust_pwm_min = self.get_parameter('low_level_controller.thrust_pwm_min').value
    self.low_level_controller_thrust_pwm_max = self.get_parameter('low_level_controller.thrust_pwm_max').value
    self.policy_path = self.get_parameter('policy.path').value
    self.takeoff_height = self.get_parameter('takeoff_height').value
    waypoints_flat = torch.tensor(self.get_parameter('policy.waypoints').value)
    self.waypoints = waypoints_flat.view(-1, 6)

    # Print parameters
    self.get_logger().info(f'c1: {self.low_level_controller_c1}')
    self.get_logger().info(f'c2: {self.low_level_controller_c2}')
    self.get_logger().info(f'c3: {self.low_level_controller_c3}')
    self.get_logger().info(f'gate_side: {self.gate_side}')
    self.get_logger().info(f'thrust_pwm_min: {self.low_level_controller_thrust_pwm_min}')
    self.get_logger().info(f'thrust_pwm_max: {self.low_level_controller_thrust_pwm_max}')
    self.get_logger().info(f'policy_path: {self.policy_path}')
    self.get_logger().info(f'policy_waypoints: {self.waypoints}')
    self.get_logger().info(f'takeoff_height: {self.takeoff_height}')

    #
    self.waypoints_quat = torch.zeros(self.waypoints.shape[0], 4, device=self.device)

    for i, waypoint_data in enumerate(self.waypoints):
        euler_angles_tensor = waypoint_data[3:6]
        euler_np = euler_angles_tensor.cpu().numpy()
        rot_from_euler = R.from_euler('xyz', euler_np)
        self.waypoints_quat[i, :] = torch.tensor(rot_from_euler.as_quat(scalar_first=True), device=self.device, dtype=torch.float32)