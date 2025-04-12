def init_parameters(self):
    """
    Init parameters
    """
    # Declare parameters
    self.declare_parameters(
        namespace='',
        parameters=[('crazyflie_names', ['']),
                    ('policy.path', ''),
                    ])

    # Get parameters
    self.crazyflie_names = self.get_parameter('crazyflie_names').value
    self.policy_path = self.get_parameter('policy.path').value

    # Print parameters
    self.get_logger().info(f'crazyflie_names: {self.crazyflie_names}')
    self.get_logger().info(f'policy_path: {self.policy_path}')
