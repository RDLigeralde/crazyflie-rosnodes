def init_parameters(self):
    """
    Init parameters
    """
    # Declare parameters
    self.declare_parameters(
        namespace='',
        parameters=[
            ('crazyflie_names', ['']),
            ('crazyradio_uris', ['']),
            ('logger_period_ms', 0),
            ('reconnection_period_ms', 0),
            ('mellinger_enable', False),
        ])

    # Get parameters
    self.crazyflie_names = self.get_parameter('crazyflie_names').value
    self.crazyradio_uris = self.get_parameter('crazyradio_uris').value
    self.logger_period_ms = self.get_parameter('logger_period_ms').value
    self.reconnection_period_ms = self.get_parameter('reconnection_period_ms').value
    # Gates whether ctrlRace.mellingerEnable is set to 1 at connect time —
    # see reconnect_clbk. Off by default: only set True when flying a
    # action_type="mellinger" checkpoint. Both this path and the
    # attitude-mixer path (actChanEnable) now run on the SAME
    # examples/app_race_policy firmware build — race_controller.c arbitrates
    # between them (and PID) at runtime — so this no longer implies a
    # different firmware image, just a different ctrlRace param.
    self.mellinger_enable = self.get_parameter('mellinger_enable').value

    # Print parameters
    self.get_logger().info('crazyflie_names: %s' % self.crazyflie_names)
    self.get_logger().info('crazyradio_uris: %s' % self.crazyradio_uris)
    self.get_logger().info('logger_period_ms: %s' % self.logger_period_ms)
    self.get_logger().info('reconnection_period_ms: %s' % self.reconnection_period_ms)
    self.get_logger().info('mellinger_enable: %s' % self.mellinger_enable)