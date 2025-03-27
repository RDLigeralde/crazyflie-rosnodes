from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from jirl_interfaces.msg import CommandCTBR

import cflib.crtp
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig

from .crazyradio_driver_qos import qos_best_effort, qos_reliable

class CrazyradioDriverNode(Node):

    # Import methods
    from .crazyradio_driver_params import init_parameters
    from .crazyradio_driver_callbacks import cmd_clbk, reconnect_clbk #, logger_clbk

    scf_dict = {}

    def __init__(self):
        super().__init__('crazyradio_driver')

        self.init_parameters()
        self.init_crazyflie()
        self.init_callback_groups()
        self.init_timers()
        self.init_subscriptions()

        self.get_logger().info('Node initialized')

    def cleanup(self):
        # self.sync_logger.disconnect()
        for scf in self.scf_dict.values():
            scf.close_link()

    def init_crazyflie(self):
        """
        Init crazyflie
        """
        cflib.crtp.init_drivers()

        # # Init logger
        # lg = LogConfig(name='Logger', period_in_ms=self.logger_period_ms)
        # lg.add_variable('pm.vbat', 'float')
        # self.sync_logger = SyncLogger(self.scf, lg)
        # self.sync_logger.connect()

    def init_callback_groups(self):
        """
        Init callback groups
        """
        # Subscribers
        self.cmd_cgroup = MutuallyExclusiveCallbackGroup()

        # Timers
        self.logger_cgroup = MutuallyExclusiveCallbackGroup()
        self.reconnect_cgroup = MutuallyExclusiveCallbackGroup()

    def init_subscriptions(self):
        """
        Init subscriptions
        """
        # CTBR command
        self.cmd_sub = self.create_subscription(
            CommandCTBR,
            '/ctbr_cmd',
            self.cmd_clbk,
            qos_best_effort,
            callback_group=self.cmd_cgroup
        )

    def init_timers(self):
        """
        Init timers
        """
        # self.logger_timer = self.create_timer(
        #     self.logger_period_ms / 1000,
        #     self.logger_clbk,
        #     callback_group=self.logger_cgroup
        # )

        # Reconnection
        self.reconnect_timer = self.create_timer(
            self.reconnection_period_ms / 1000,
            self.reconnect_clbk,
            callback_group=self.reconnect_cgroup
        )