from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from jirl_interfaces.msg import CommandCTBR, CommandAction, CommandAttitude, OdometryArray
from jirl_interfaces.srv import Arm

from cflib.utils import uri_helper

import cflib.crtp
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

from .crazyradio_driver_qos import qos_best_effort, qos_reliable

class CrazyradioDriverNode(Node):

    # Import methods
    from .crazyradio_driver_params import init_parameters
    from .crazyradio_driver_callbacks import cmd_clbk, reconnect_clbk, mocap_clbk, arm_clbk, action_clbk, attitude_clbk #, logger_clbk

    scf_dict = {}

    def __init__(self):
        super().__init__('crazyradio_driver')

        self.init_parameters()
        self.init_crazyflie()
        self.init_callback_groups()
        #self.init_timers()
        self.init_subscriptions()
        self.init_services()

        for crazyradio_uri, crazyflie_name in zip(self.crazyradio_uris, self.crazyflie_names):
            if crazyflie_name in self.scf_dict:
                connected = self.scf_dict[crazyflie_name].is_link_open()
                if not connected:
                    self.get_logger().error('Crazyflie %s disconnected' % (crazyflie_name))
                    self.scf_dict.pop(crazyflie_name)
            else:
                self.get_logger().warn('Trying to connect to Crazyflie %s...' % crazyflie_name)
                try:
                    URI = uri_helper.uri_from_env(default=crazyradio_uri)

                    self.scf_dict[crazyflie_name] = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
                    self.scf_dict[crazyflie_name].open_link()

                    self.get_logger().warn('Sending zero command to %s' % crazyradio_uri)
                    self.scf_dict[crazyflie_name].cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
                    self.get_logger().warn('Connected to %s' % crazyradio_uri)

                    self.scf_dict[crazyflie_name].cf.param.set_value('stabilizer.estimator', '2')
                    self.scf_dict[crazyflie_name].param.set_value('locSrv.extQuatStdDev', 0.06)
                    if self.mellinger_enable:
                        # See crazyradio_driver_callbacks.py's reconnect_clbk
                        # for the full explanation — sets ctrlRace.mellingerEnable,
                        # not stabilizer.controller.
                        self.scf_dict[crazyflie_name].cf.param.set_value('ctrlRace.mellingerEnable', '1')
                except Exception as e:
                    continue

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
        self.mocap_cgroup = MutuallyExclusiveCallbackGroup()

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

        # Mocap data
        self.mocap_sub = self.create_subscription(
            OdometryArray,
            '/multi_odometry',
            self.mocap_clbk,
            qos_best_effort,
            callback_group=self.mocap_cgroup
        )

        # Custom controller / open-loop mixer path (JaxRacingPolicy
        # action_type="attitude" — see controller_utils.py's
        # send_action_command). Parallel to /ctbr_cmd's own subscription
        # above, same callback group since it's the same rate-class of
        # per-tick traffic to the same Crazyflie, sent instead of it (not
        # alongside) when active.
        self.action_sub = self.create_subscription(
            CommandAction,
            '/action_cmd',
            self.action_clbk,
            qos_best_effort,
            callback_group=self.cmd_cgroup
        )

        # Onboard-Mellinger path (JaxRacingPolicy action_type="mellinger"
        # — see controller_utils.py's send_attitude_command). Also sent
        # instead of /ctbr_cmd, not alongside.
        self.attitude_sub = self.create_subscription(
            CommandAttitude,
            '/attitude_cmd',
            self.attitude_clbk,
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

    def init_services(self):
        """
        Init services
        """

        # Arm or Disarm
        self.arm_srv = self.create_service(
            Arm,
            '/arm',
            self.arm_clbk)