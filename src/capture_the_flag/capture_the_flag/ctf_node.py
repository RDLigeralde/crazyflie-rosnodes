from threading import Lock

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nav_msgs.msg import Odometry
from jirl_interfaces.msg import Setpoint

from .ctf_qos import *

import torch
import torch.nn as nn

class CTF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CTF, self).__init__()

        self.ctf = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.ctf(x)

class CaptureTheFlagNode(Node):

    # Import methods
    from .ctf_params import init_parameters
    from .ctf_callbacks import mocap_clbk
    # from .ctf_utils import

    mocap_lock = Lock()
    traj_lock = Lock()
    mocap_pose = {}
    device = torch.device('cpu')

    def __init__(self):
        super().__init__('capture_the_flag')

        self.init_parameters()
        self.init_publishers()
        self.init_NN()
        self.init_callback_groups()
        self.init_subscriptions()

        self.get_logger().info('Node initialized')

    def cleanup(self):
        pass

    def init_NN(self):
        """
        Init neural network
        """
        # Create network
        self.model = CTF(2*len(self.crazyflie_names), 2).to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.policy_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint, strict=False)

    def init_callback_groups(self):
        """
        Init callback groups
        """
        # Subscribers
        self.mocap_cgroup = MutuallyExclusiveCallbackGroup()

    def init_publishers(self):
        """
        Init publishers
        """
        # Setpoint
        self.publishers = {}
        for name in self.crazyflie_names:
            self.publishers[name] = self.create_publisher(
                Setpoint,
                f'/{name}/setpoint',
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
