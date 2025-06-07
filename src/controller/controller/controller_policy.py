import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

class ScalarFiLM(nn.Module):
    def __init__(self, cond_dim, film_hidden_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, film_hidden_dims[0]),
            nn.GELU(),
            nn.Linear(film_hidden_dims[0], film_hidden_dims[1]),
            nn.GELU(),
            nn.Linear(film_hidden_dims[1], 2)
        )

    def forward(self, x, cond):
        gamma_beta = self.mlp(cond)
        gamma, beta = gamma_beta[0], gamma_beta[1]
        return gamma * x + beta

class FiLMActor(nn.Module):
    def __init__(self, mlp_input_dim, actor_hidden_dims, num_actions, cond_dim, film_hidden_dims, activation):
        super().__init__()

        self.activation = activation()

        self.actor = nn.Module()
        self.actor.fc1 = nn.Linear(mlp_input_dim, actor_hidden_dims[0])
        self.actor.hidden_layers = nn.ModuleList()
        for i in range(1, len(actor_hidden_dims)):
            self.actor.hidden_layers.append(nn.Linear(actor_hidden_dims[i - 1], actor_hidden_dims[i]))
        self.actor.output_layer = nn.Linear(actor_hidden_dims[-1], num_actions)
        self.tanh = nn.Tanh()

        self.actor.film = ScalarFiLM(cond_dim, film_hidden_dims)
        self.actor.film.eval()
        self.cond_dim = cond_dim

    def forward(self, obs):
        x = self.activation(self.actor.fc1(obs))
        x = self.actor.film(x, obs[-self.cond_dim:])

        for layer in self.actor.hidden_layers:
            x = self.activation(layer(x))
        x = self.tanh(self.actor.output_layer(x))
        return x

class RacingPolicy:
    def __init__(self, vehicle, model_path, waypoints, waypoints_quat, gate_side, scale_output=True, device="cpu"):
        self.quadrotor = vehicle
        self.device = torch.device(device)
        self.obs_dim = 3 + 9 + 12 + 12 + 2

        self.action_dim = 4

        self.scale_output = scale_output

        self.waypoints = waypoints
        self.waypoints_quat = waypoints_quat

        self.gate_side = gate_side
        d = gate_side / 2
        self.local_square = torch.tensor([
            [0,  d,  d],
            [0, -d,  d],
            [0, -d, -d],
            [0,  d, -d]
        ], dtype=torch.float32, device=self.device)

        # Create network
        self.model = FiLMActor(self.obs_dim, [128, 128], self.action_dim, 2, [3,3], nn.ELU).to(self.device)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        actor_state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if "actor" in k}
        self.model.load_state_dict(actor_state_dict, strict=True)

        self.model.eval()

        ######  Min/max values for scaling control outputs.
        rotor_speed_max = self.quadrotor['rotor_speed_max']
        rotor_speed_min = self.quadrotor['rotor_speed_min']

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.quadrotor['k_eta'] * rotor_speed_max**2
        self.min_thrust = self.quadrotor['k_eta'] * rotor_speed_min**2

        # Set the maximum body rate on each axis (this is hand selected), rad/s
        self.max_roll_br = 2.0
        self.max_pitch_br = 2.0
        self.max_yaw_br = 1.0

        self.idx_wp = 0

        self.cond_twr = torch.tensor([1.2])
        self.cond_perc = torch.tensor([0.0])

    def update(self, state):
        """
        Compute the control command using the neural network.

        Inputs:
            t, current time in seconds
            state, current state with keys:
                - x: absolute position (3,)
                - v: linear velocity (3,)
                - q: quaternion [i, j, k, w]
                - w: angular velocity (3,)
            flat_output, desired output with keys:
                - x: target position (3,)
         Output:
            control_input, dictionary with 4 controls:
                - cmd_thrust
                - cmd_moment
        """
        pos_drone = torch.tensor(state['x'], dtype=torch.float32, device=self.device)
        lin_vel_drone = torch.tensor(state['v_b'], dtype=torch.float32, device=self.device)
        quat_drone = torch.tensor(state['q'], dtype=torch.float32, device=self.device)
        rot_drone = torch.tensor(R.from_quat(quat_drone).as_matrix(), dtype=torch.float32, device=self.device)

        curr_idx = self.idx_wp
        next_idx = (self.idx_wp + 1) % self.waypoints.shape[0]

        wp_curr_pos = self.waypoints[curr_idx, :3]
        wp_next_pos = self.waypoints[next_idx, :3]
        quat_curr = self.waypoints_quat[curr_idx, :]
        quat_next = self.waypoints_quat[next_idx, :]
        rot_curr = torch.tensor(R.from_quat(quat_curr).as_matrix(), dtype=torch.float32, device=self.device)
        rot_next = torch.tensor(R.from_quat(quat_next).as_matrix(), dtype=torch.float32, device=self.device)

        pose_drone_wrt_gate = self._subtract_frame_transforms(wp_curr_pos, rot_curr, pos_drone)

        if torch.norm(pose_drone_wrt_gate) < self.gate_side and pose_drone_wrt_gate[0] < 0.10:
            self.idx_wp = (self.idx_wp + 1) % self.waypoints.shape[0]

        verts_curr = self.local_square @ rot_curr.T + wp_curr_pos.unsqueeze(0)
        verts_next = self.local_square @ rot_next.T + wp_next_pos.unsqueeze(0)

        waypoint_pos_b_curr = self._subtract_frame_transforms(pos_drone, rot_drone, verts_curr)
        waypoint_pos_b_next = self._subtract_frame_transforms(pos_drone, rot_drone, verts_next)

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(4, 3)

        obs = torch.cat(
            [
                lin_vel_drone.flatten(),
                rot_drone.flatten(),
                waypoint_pos_b_curr.flatten(),
                waypoint_pos_b_next.flatten(),
                self.cond_twr.flatten(),
                self.cond_perc.flatten()
            ],
            dim=-1,
        )
        print(obs[0:3])
        print(obs[3:12])
        print()
        print(obs[12:15])
        print(obs[15:18])
        print(obs[18:21])
        print(obs[21:24])
        print()
        print(obs[24:27])
        print(obs[27:30])
        print(obs[30:33])
        print(obs[33:36])
        print()
        print(obs[36:38])
        print()
        print()
        print(wp_curr_pos)

        actions = self.model(obs).squeeze(0).detach().cpu().numpy()
        print(actions)
        actions = np.clip(actions, -1, 1)

        num_rotors = self.quadrotor['num_rotors']
        cmd_thrust = actions[0]
        cmd_w = np.array([actions[1], actions[2], actions[3]])

        if self.scale_output:
            cmd_thrust = np.interp(cmd_thrust,
                                   [-1, 1],
                                   [num_rotors * self.min_thrust, num_rotors * self.max_thrust])

            cmd_w[0] = np.interp(cmd_w[0],
                                 [-1, 1],
                                 [-self.max_roll_br, self.max_roll_br])
            cmd_w[1] = np.interp(cmd_w[1],
                                 [-1, 1],
                                 [-self.max_pitch_br, self.max_pitch_br])
            cmd_w[2] = np.interp(cmd_w[2],
                                 [-1, 1],
                                 [-self.max_yaw_br, self.max_yaw_br])

        control_input = {'cmd_motor_speeds': np.zeros((4,)),
                         'cmd_motor_thrusts': np.zeros((4,)),
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': np.zeros((3,)),
                         'cmd_q': np.array([0, 0, 0, 1]),
                         'cmd_w': cmd_w * 180 / np.pi,
                         'cmd_v': np.zeros((3,))}

        return control_input

    def _subtract_frame_transforms(self, pos, rot, pos_des):
        if pos_des.ndim == 1:
            return rot.T @ (pos_des - pos)
        elif pos_des.ndim == 2:
            return (pos_des - pos) @ rot
