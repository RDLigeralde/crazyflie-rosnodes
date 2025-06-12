from threading import Lock

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from jirl_interfaces.msg import CommandCTBR, Trajectory, Observations
from jirl_interfaces.srv import UpdateSetpoint, StartTrajectory

from rotorpy.controllers.quadrotor_control import SE3ControlCTBR

from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params

import torch

from .controller_qos import *
from .controller_fsm import ControllerFSM
from .controller_policy import RacingPolicy

class ControllerNode(Node):

    # Import methods
    from .controller_params import init_parameters
    from .controller_callbacks import mocap_clbk, logger_clbk, update_setpoint_clbk, landing_clbk, takeoff_clbk, trajectory_clbk, race_clbk
    from .controller_utils import send_ctbr_command, send_trajectory

    mocap_lock = Lock()
    traj_lock = Lock()
    mocap_pose = {}
    device = torch.device('cpu')

    def __init__(self):
        super().__init__('controller')

        self.init_parameters()
        self.init_fsm()
        self.init_publishers()
        self.init_controller()
        self.init_callback_groups()
        self.init_services()
        self.init_subscriptions()

        self.get_logger().info('Node initialized')

    def cleanup(self):
        pass

    def init_fsm(self):
        """
        Init FSM
        """
        self.fsm = ControllerFSM()

    def print_state(self):
        self.get_logger().info(f'Entering state: {self.fsm.state}')

    def init_controller(self):
        """
        Init controller
        """
        self.policy = RacingPolicy(crazyflie_params, self.policy_path, self.waypoints, self.waypoints_quat, self.gate_side, device=self.device)
        self.se3_controller = SE3ControlCTBR(crazyflie_params)

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
        # CTBR command
        if self.crazyradio_driver == 'cpp':
            topic_name = 'ctbr_cmd'
        elif self.crazyradio_driver in ['py', 'python']:
            topic_name = '/ctbr_cmd'
        self.cmd_pub = self.create_publisher(
            CommandCTBR,
            topic_name,
            qos_best_effort
        )

        # Trajectory data
        self.traj_pub = self.create_publisher(
            Trajectory,
            'trajectory',
            qos_best_effort
        )

        # Observations
        self.obs_pub = self.create_publisher(
            Observations,
            'observations',
            qos_best_effort
        )

    def init_subscriptions(self):
        """
        Init subscriptions
        """
        # Mocap odometry
        self.mocap_sub = self.create_subscription(
            Odometry,
            '/mocap',
            self.mocap_clbk,
            qos_best_effort,
            callback_group=self.mocap_cgroup
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

        # StartTrajectory
        self.trajectory_srv = self.create_service(
            StartTrajectory,
            'trajectory',
            self.trajectory_clbk)

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

        # Start racing
        self.racing_srv = self.create_service(
            Trigger,
            'race',
            self.race_clbk)