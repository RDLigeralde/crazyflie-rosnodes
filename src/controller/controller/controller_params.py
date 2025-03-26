def init_parameters(self):
    """
    Init parameters
    """
    # Declare parameters
    self.declare_parameters(
        namespace='',
        parameters=[('low_level_controller.c1', 0.0),
                    ('low_level_controller.c2', 0.0),
                    ('low_level_controller.c3', 0.0),
                    ('low_level_controller.thrust_pwm_min', 0),
                    ('low_level_controller.thrust_pwm_max', 0),
                    ('policy.enable', False),
                    ('policy.path', ''),
                    ('takeoff_height', 0.5),
                    ])

    # Get parameters
    self.low_level_controller_c1 = self.get_parameter('low_level_controller.c1').value
    self.low_level_controller_c2 = self.get_parameter('low_level_controller.c2').value
    self.low_level_controller_c3 = self.get_parameter('low_level_controller.c3').value
    self.low_level_controller_thrust_pwm_min = self.get_parameter('low_level_controller.thrust_pwm_min').value
    self.low_level_controller_thrust_pwm_max = self.get_parameter('low_level_controller.thrust_pwm_max').value
    self.policy_enabled = self.get_parameter('policy.enable').value
    self.policy_path = self.get_parameter('policy.path').value
    self.takeoff_height = self.get_parameter('takeoff_height').value

    # Print parameters
    self.get_logger().info(f'c1: {self.low_level_controller_c1}')
    self.get_logger().info(f'c2: {self.low_level_controller_c2}')
    self.get_logger().info(f'c3: {self.low_level_controller_c3}')
    self.get_logger().info(f'thrust_pwm_min: {self.low_level_controller_thrust_pwm_min}')
    self.get_logger().info(f'thrust_pwm_max: {self.low_level_controller_thrust_pwm_max}')
    self.get_logger().info(f'policy_enabled: {self.policy_enabled}')
    self.get_logger().info(f'policy_path: {self.policy_path}')
    self.get_logger().info(f'takeoff_height: {self.takeoff_height}')
