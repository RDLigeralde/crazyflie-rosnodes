import numpy as np
from numpy import pi, ceil
import re
from threading import Thread, Lock

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray
from jirl_interfaces.srv import UpdateSetpoint

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

import torch

from .controller_qos import qos_best_effort, qos_reliable

device = torch.device('cpu')
pos_des = [1.0, 3.0, 0.5]

class ControllerNode(Node):

    # Import methods
    from .controller_params import init_parameters
    from .controller_callbacks import mocap_clbk, cmd_clbk, update_setpoint_clbk, landing_clbk, takeoff_clbk
    # from .controller_utils import

    def __init__(self):
        super().__init__('controller')

        self.init_parameters()
        self.init_publishers()
        self.init_crazyflie()
        self.init_callback_groups()
        self.init_services()
        self.init_timers()
        self.init_subscriptions()

        self.get_logger().info('Node initialized')

    def cleanup(self):
        self.scf.close_link()

    def init_crazyflie(self):
        """
        Init crazyflie
        """
        self.state = 'flying'
        self.mocap_pose = {}

        self.controller = SE3Control(crazyflie_params)
        self.trajectory = HoverTraj(x0=pos_des)

        cflib.crtp.init_drivers()

        URI = uri_helper.uri_from_env(default=self.crazyradio_uri)
        self.scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
        self.scf.open_link()

    def init_callback_groups(self):
        """
        Init callback groups
        """
        # Subscribers
        self.mocap_cgroup = MutuallyExclusiveCallbackGroup()

        # Timers
        self.cmd_cgroup = MutuallyExclusiveCallbackGroup()

    def init_publishers(self):
        """
        Init publishers
        """
        # # Estimated pose
        # self.pose_pub = self.create_publisher(
        #     PoseWithCovarianceStamped,
        #     '/estimated_pose',
        #     qos_best_effort
        # )

    def init_subscriptions(self):
        """
        Init subscriptions
        """
        # Mocap odometry
        self.mocap_sub = self.create_subscription(
            Odometry,
            '/mocap',
            self.mocap_clbk,
            qos_best_effort
        )

    def init_timers(self):
        """
        Init timers
        """
        self.cmd_timer = self.create_timer(
            1 / self.control_freq,
            lambda: self.cmd_clbk(),
            callback_group=self.cmd_cgroup
        )

    def init_services(self):
        """
        Init services
        """
        # Setpoint update
        self.update_setpoint_srv = self.create_service(
            UpdateSetpoint,
            'update_setpoint',
            self.update_setpoint_clbk)

        # Landing
        self.land_srv = self.create_service(
            Trigger,
            'land',
            self.landing_clbk)

        # Takeoff
        self.takeoff_srv = self.create_service(
            Trigger,
            'takeoff',
            self.takeoff_clbk)