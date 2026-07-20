"""JAX/Flax policy loader for checkpoints produced by mjc_dronetests/train.py
(the sibling MJX/PureJaxRL race-training repo), as an alternative to
controller_policy.py's PyTorch RacingPolicy.

Same public interface as RacingPolicy (same update(state) -> (control, obs)
contract, consumed identically by controller_utils.py's single_update), so
selecting this policy instead is a drop-in swap gated by a ROS param — see
controller_params.py's policy_jax.* / controller_node.py's init_controllers.

IMPORTANT — things ported/verified vs. things requiring field calibration:

  * The v3 observation layout and the gate-crossing state machine below are
    a byte-for-byte port of mjc_dronetests' race/observations.py's _V3 and
    race/plugin.py's RacePlugin.step(), cross-validated numerically against
    a live MJX rollout (max diff ~2e-7, float-precision noise) before this
    file was written. The vendored ActorCritic/ActorCriticRNN network
    classes below were likewise verified to produce bit-identical output
    to mjc_dronetests/models.py's real networks given the same params.pkl.
    These parts are NOT guesses.

  * The roll/pitch/yaw_rate -> degrees/second scale factors
    (max_roll_rate_dps/max_pitch_rate_dps/max_yaw_rate_dps) are NOT
    analytically derivable from the simulator: the trained policy's raw
    action[1:4] outputs are fixed-scale per-motor RPM *differentials* in
    the sim's mixer (see mix_attitude_rpm's docstring in dmcdrones) — an
    open-loop torque-like proxy, not a physical angular rate in any
    standard unit — whereas the real Crazyflie's CTBR setpoint expects a
    genuine physical body rate in degrees/second. There is no unit
    conversion that connects the two; the scale is a real calibration
    constant that must be tuned on hardware (start conservative and small,
    verify behavior at low thrust/tethered before increasing). The defaults
    below are placeholders, not derived values.

  * cmd_thrust, by contrast, IS analytically correct: it reproduces the
    sim's own hover_rpm/max_rpm two-segment mapping (see mix_rpm_action's
    docstring in dmcdrones) using the exact hover_rpm/max_rpm/kf this
    checkpoint was trained with (config.json), rather than assuming hover
    sits at the midpoint of the PWM range.

  * Per your explicit direction, this only supports action_type="attitude"
    and obs_fn="v3" (raises a clear error otherwise) — race/ only, matching
    the current race_mjx.yml / race_recurrent_mjx.yml training configs.
"""

import functools
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


# ---------------------------------------------------------------------------
# Vendored networks — must stay structurally identical to mjc_dronetests/
# models.py's ActorCritic/ActorCriticRNN (same Dense-layer declaration order,
# same nested-module CLASS NAMES — flax's param paths include submodule
# class names, not just declaration order, confirmed by testing) for a
# params.pkl trained there to load correctly here. Critic head and the
# distrax distribution construction are both omitted (control uses the
# deterministic actor mean, never a stochastic sample or a value estimate),
# which requires strictly fewer params than the saved checkpoint has —
# verified this is safe: flax's apply() only reads params this module's
# __call__ actually looks up by name; extra keys in the loaded dict
# (log_std, critic Dense_*) are simply ignored, not an error.
# ---------------------------------------------------------------------------

def _activation_fn(name: str):
    if name == "tanh":
        return nn.tanh
    elif name == "relu":
        return nn.relu
    return nn.elu


class ActorCritic(nn.Module):
    action_dim: int
    hidden: Sequence[int] = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = _activation_fn(self.activation)
        actor_mean = x
        for h in self.hidden:
            actor_mean = nn.Dense(
                h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(actor_mean)
            actor_mean = act(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        return actor_mean


class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(ins.shape[0], self.hidden_size),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: int
    hidden: Sequence[int] = (256,)
    rnn_hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, hstate, x):
        obs, resets = x
        act = _activation_fn(self.activation)
        embedding = obs
        for h in self.hidden:
            embedding = nn.Dense(
                h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(embedding)
            embedding = act(embedding)
        new_hstate, embedding = ScannedRNN(hidden_size=self.rnn_hidden_size)(
            hstate, (embedding, resets)
        )
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        return new_hstate, actor_mean

    def initialize_carry(self, batch_size):
        return ScannedRNN.initialize_carry(batch_size, self.rnn_hidden_size)


# ---------------------------------------------------------------------------
# mix_rpm_action's two-segment thrust mapping, inverted: given a desired
# per-motor RPM (from hover_rpm/max_rpm below), recover the normalized
# [0, 1] fraction of the *raw PWM range* controller_utils.py's
# thrust_pwm = thrust_pwm_min + frac*(thrust_pwm_max - thrust_pwm_min)
# expects — NOT simply action[0] itself, since hover_rpm is not the
# midpoint of [0, max_rpm]. See dmcdrones' utils/mixer.py:mix_rpm_action.
# ---------------------------------------------------------------------------

def _thrust_action_to_pwm_frac(thrust_action: float, hover_rpm: float, max_rpm: float) -> float:
    if thrust_action <= 0.0:
        rpm = (thrust_action + 1.0) * hover_rpm
    else:
        rpm = hover_rpm + (max_rpm - hover_rpm) * thrust_action
    return float(np.clip(rpm / max_rpm, 0.0, 1.0))


class JaxRacingPolicy:
    """Loads a mjc_dronetests checkpoint (run_dir/{params.pkl,config.json})
    and runs it in an open-loop-attitude (CTBR) control loop, matching
    RacingPolicy's update(state) -> (control, obs) interface exactly.

    params (dict) mirrors RacingPolicy's own constructor contract:
        "gate_side": float
        "max_roll_rate_dps" / "max_pitch_rate_dps" / "max_yaw_rate_dps": float
            — REQUIRES HARDWARE CALIBRATION, see module docstring.
    Gate positions/normals come from run_dir/config.json (saved by
    mjc_dronetests/train.py from the actual gates the checkpoint trained
    against), not from a separately-configured waypoints param — avoids the
    two ever silently drifting apart.
    """

    def __init__(self, run_dir, params: dict, device: str = "cpu"):
        run_dir = Path(run_dir)
        with open(run_dir / "config.json") as f:
            cfg = json.load(f)

        if cfg["task"] != "race":
            raise ValueError(f"JaxRacingPolicy only supports task='race' checkpoints, got {cfg['task']!r}")
        if cfg["action_type"] != "attitude":
            raise ValueError(
                f"JaxRacingPolicy only supports action_type='attitude' (open-loop CTBR) "
                f"checkpoints, got {cfg['action_type']!r}"
            )
        if cfg["obs_fn"] != "v3":
            raise ValueError(
                f"JaxRacingPolicy only supports obs_fn='v3' checkpoints (the only "
                f"observation layout ported/verified here), got {cfg['obs_fn']!r}"
            )

        self.action_dim = cfg["action_dim"]
        self.recurrent = bool(cfg.get("recurrent", False))
        self.hover_rpm = float(cfg["hover_rpm"])
        self.max_rpm = float(cfg["max_rpm"])

        self.gate_positions = np.asarray(cfg["gate_positions"], dtype=np.float64)  # (N, 3)
        self.gate_normals = np.asarray(cfg["gate_normals"], dtype=np.float64)      # (N, 3)
        self.n_gates = self.gate_positions.shape[0]
        self.gate_side = params.get("gate_side", cfg.get("gate_side", 1.0))
        self._half_gate = self.gate_side / 2.0

        # Gate-tracking state — mirrors race/plugin.py's RacePlugin exactly
        # (gate_idx, prev_gate_x; see _update_gate_tracking below). Starts
        # targeting gate 0 with prev_gate_x=1.0 (matches init_task_state's
        # convention: no prior crossing, positioned on the approach side).
        self.gate_idx = 0
        self.prev_gate_x = 1.0

        if self.recurrent:
            self.network = ActorCriticRNN(
                action_dim=self.action_dim,
                hidden=tuple(cfg["hidden"]),
                rnn_hidden_size=cfg.get("rnn_hidden_size", 128),
                activation=cfg["activation"],
            )
            self.hstate = self.network.initialize_carry(1)
            # last_done marks whether the *previous* control cycle's obs was
            # itself a fresh reset — see ppo_rnn.py's module docstring for
            # why this must be the previous step's, not this step's, done
            # flag. There is no episode boundary on real hardware during a
            # single continuous racing run, so this is always False after
            # the first call — reset() below is provided for a new attempt.
            self.last_done = True
        else:
            self.network = ActorCritic(
                action_dim=self.action_dim,
                hidden=tuple(cfg["hidden"]),
                activation=cfg["activation"],
            )

        with open(run_dir / "params.pkl", "rb") as f:
            import pickle
            self.net_params = pickle.load(f)

        # Warm-up / shape-check the network once at load time so a shape
        # mismatch surfaces immediately at startup, not on the first control
        # cycle in flight.
        dummy_obs = jnp.zeros((21,), dtype=jnp.float32)
        if self.recurrent:
            _, _ = self.network.apply(
                self.net_params, self.hstate,
                (dummy_obs[jnp.newaxis, jnp.newaxis, :], jnp.array([[True]])),
            )
        else:
            _ = self.network.apply(self.net_params, dummy_obs)

        self.max_roll_rate_dps = params.get("max_roll_rate_dps", 90.0)
        self.max_pitch_rate_dps = params.get("max_pitch_rate_dps", 90.0)
        self.max_yaw_rate_dps = params.get("max_yaw_rate_dps", 90.0)

    def reset(self):
        """Call at the start of a new racing attempt — resets gate-tracking
        state and (for a recurrent checkpoint) the GRU hidden state."""
        self.gate_idx = 0
        self.prev_gate_x = 1.0
        if self.recurrent:
            self.hstate = self.network.initialize_carry(1)
            self.last_done = True

    def _update_gate_tracking(self, pos: np.ndarray, R: np.ndarray) -> None:
        """Byte-for-byte port of race/plugin.py's RacePlugin.step() gate-
        crossing test — verified against a live MJX rollout before this file
        was written (see module docstring)."""
        gate_pos = self.gate_positions[self.gate_idx]
        normal = self.gate_normals[self.gate_idx]
        rel = pos - gate_pos
        x_wrt_gate = float(rel @ normal)
        y_wrt_gate = float(rel[1] * normal[0] - rel[0] * normal[1])
        z_wrt_gate = float(rel[2])
        just_passed = (
            x_wrt_gate < 0.0
            and self.prev_gate_x > 0.0
            and abs(y_wrt_gate) < self._half_gate
            and abs(z_wrt_gate) < self._half_gate
        )
        if just_passed:
            self.gate_idx = (self.gate_idx + 1) % self.n_gates
            self.prev_gate_x = 1.0
        else:
            self.prev_gate_x = x_wrt_gate

    def _build_obs(self, pos: np.ndarray, R: np.ndarray, lin_vel_b: np.ndarray, ang_vel_b: np.ndarray) -> np.ndarray:
        """Byte-for-byte port of race/observations.py's _V3 — verified
        against a live MJX rollout before this file was written (see module
        docstring)."""
        next_idx = (self.gate_idx + 1) % self.n_gates
        gravity_b = R[2, :]
        target_b = R.T @ (self.gate_positions[self.gate_idx] - pos)
        target_b_next = R.T @ (self.gate_positions[next_idx] - pos)
        normal_b = R.T @ self.gate_normals[self.gate_idx]
        normal_b_next = R.T @ self.gate_normals[next_idx]
        return np.concatenate([
            lin_vel_b, ang_vel_b, gravity_b, target_b, target_b_next, normal_b, normal_b_next,
        ]).astype(np.float32)

    def update(self, state):
        """state must provide 'x' (world position), 'R' (body->world
        rotation matrix), 'v_b' (body-frame linear velocity), 'w_b'
        (body-frame angular velocity) — all already present in
        controller_utils.py's self.mocap_pose dict."""
        pos = np.asarray(state["x"], dtype=np.float64)
        R = np.asarray(state["R"], dtype=np.float64)
        lin_vel_b = np.asarray(state["v_b"], dtype=np.float64)
        ang_vel_b = np.asarray(state["w_b"], dtype=np.float64)

        self._update_gate_tracking(pos, R)
        obs = self._build_obs(pos, R, lin_vel_b, ang_vel_b)
        obs_jax = jnp.asarray(obs)

        if self.recurrent:
            ac_in = (obs_jax[jnp.newaxis, jnp.newaxis, :], jnp.array([[self.last_done]]))
            new_hstate, actor_mean_seq = self.network.apply(self.net_params, self.hstate, ac_in)
            self.hstate = new_hstate
            self.last_done = False
            actor_mean = np.asarray(actor_mean_seq[0, 0])
        else:
            actor_mean = np.asarray(self.network.apply(self.net_params, obs_jax))

        # Deterministic action (the policy's mean, not a stochastic sample —
        # matches how a trained policy is normally deployed), clipped to
        # [-1, 1] exactly like the training-time env.step()'s own clip.
        action = np.clip(actor_mean, -1.0, 1.0)

        cmd_thrust = _thrust_action_to_pwm_frac(float(action[0]), self.hover_rpm, self.max_rpm)

        # See module docstring: these scale factors are NOT analytically
        # derived and MUST be calibrated on hardware before flight.
        roll_dps = float(action[1]) * self.max_roll_rate_dps
        pitch_dps = float(action[2]) * self.max_pitch_rate_dps
        yaw_dps = float(action[3]) * self.max_yaw_rate_dps

        control_input = {
            "cmd_thrust": cmd_thrust,
            "cmd_w": np.array([roll_dps, pitch_dps, yaw_dps]),
        }
        return control_input, obs
