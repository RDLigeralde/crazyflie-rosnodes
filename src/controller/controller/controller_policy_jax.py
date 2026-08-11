"""JAX/Flax policy loader for checkpoints produced by mjc_dronetests/train.py
(the sibling MJX/PureJaxRL race-training repo), as an alternative to
controller_policy.py's PyTorch RacingPolicy.

Same public interface as RacingPolicy (same update(state) -> (control, obs)
contract, consumed identically by controller_utils.py's single_update), so
selecting this policy instead is a drop-in swap gated by a ROS param — see
controller_params.py's policy.jax_enable / controller_node.py's
init_controllers.

Supports checkpoints trained with either of two crazyflow action_types
(read straight out of config.json — see __init__), dispatched to two
different real-hardware controllers because the two conventions expect
different things downstream:

  * action_type="attitude" — dmcdrones' fixed-gain differential mixer
    convention, raw action = [thrust_norm, roll, pitch, yaw_rate]. update()
    returns this open-loop-CTBR-ish control_input for backward
    compatibility AND the raw clipped action (control_input["raw_action"])
    for controller_utils.py to stream via the new Crazyflie::sendRaceAction
    path — crazyflie-firmware's examples/app_race_policy/src/mixer.c runs
    the SAME mixer on-device (byte-exact C port of
    crazyflow_interface/_mixer.py's mix_attitude_rpm) and bypasses PID
    entirely, so this file no longer needs its own calibrated rate-scale
    approximation of that mixer for real flight (max_roll_rate_dps etc.
    stay only as a fallback/reference, see update()).

  * action_type="mellinger" — a real physical [roll, pitch, yaw, thrust]
    attitude setpoint (radians, Newtons), rescaled exactly the way
    crazyflow_interface/__init__.py's _attitude_cmd does at train/sim time.
    Dispatched to firmware's real onboard Mellinger controller via a
    legacy RPYT setpoint (roll/pitch as real angles, yaw closed into a rate
    command by a thin on-host P-wrapper — see update()) — no on-device
    mixer involved, no crazyflow import needed here (the rescale bounds
    are persisted as plain floats in config.json by train.py, matching how
    hover_rpm/max_rpm are already persisted for the attitude path).

IMPORTANT — things ported/verified vs. things requiring field calibration:

  * The v3 observation layout and the gate-crossing state machine below are
    a byte-for-byte port of mjc_dronetests' race/observations.py's _V3 and
    race/plugin.py's RacePlugin.step(), cross-validated numerically against
    a live MJX rollout (max diff ~2e-7, float-precision noise) before this
    file was written. The vendored ActorCritic/ActorCriticRNN network
    classes below were likewise verified to produce bit-identical output
    to mjc_dronetests/models.py's real networks given the same params.pkl.
    These parts are NOT guesses.

  * action_type="attitude"'s max_roll_rate_dps/max_pitch_rate_dps/
    max_yaw_rate_dps are NOT analytically derivable from the simulator (see
    the old design note this file used to lead with) and are kept only as
    an emergency-fallback control_input shape — real flight should use
    control_input["raw_action"] via sendRaceAction, not these. cmd_thrust
    IS analytically correct: it reproduces the sim's own hover_rpm/max_rpm
    two-segment mapping using the exact hover_rpm/max_rpm this checkpoint
    was trained with (config.json).

  * action_type="mellinger"'s roll/pitch/yaw/thrust bounds ARE analytically
    correct (a straight port of _attitude_cmd's own rescale, not a
    calibration guess) — what still needs on-hardware tuning there is only
    the yaw P-wrapper's gain (self.yaw_kp) and, as with any first flight,
    verifying action=0 actually reproduces real hover.

  * Only obs_fn="v3" is supported (raises a clear error otherwise) — race/
    only, matching the current race_mjx.yml / race_recurrent_mjx.yml
    training configs. ma_race (multi-agent) checkpoints are out of scope
    for this loader.
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
    """Loads a mjc_dronetests checkpoint from its config.json (the sibling
    params.pkl, if present, is loaded lazily — see below) and runs it,
    matching RacingPolicy's update(state) -> (control, obs) interface
    exactly. Dispatch shape of the returned control_input depends on
    self.action_type ("attitude" or "mellinger") — see module docstring
    and update() below.

    params (dict) mirrors RacingPolicy's own constructor contract:
        "gate_side": float
        "max_roll_rate_dps" / "max_pitch_rate_dps" / "max_yaw_rate_dps": float
            — action_type="attitude" fallback control_input only, see
            module docstring. Unused for action_type="mellinger".
        "yaw_kp": float — action_type="mellinger" only, the on-host
            desired-yaw-angle -> yaw-rate P-wrapper gain (1/s per radian of
            error). Needs on-hardware tuning; start conservative.
    Gate positions/normals come from config.json (saved by
    mjc_dronetests/train.py from the actual gates the checkpoint trained
    against), not from a separately-configured waypoints param — avoids the
    two ever silently drifting apart.

    config_path is the path to config.json itself, not the run directory.
    params.pkl is loaded from alongside config_path (config_path.parent /
    "params.pkl") only if it actually exists there — when it doesn't,
    self.net_params stays None and update() raises a clear error instead of
    failing on a missing file at construction time.
    """

    def __init__(self, config_path, params: dict, device: str = "cpu"):
        config_path = Path(config_path)
        run_dir = config_path.parent
        with open(config_path) as f:
            cfg = json.load(f)

        if cfg["task"] != "race":
            raise ValueError(f"JaxRacingPolicy only supports task='race' checkpoints, got {cfg['task']!r}")
        self.action_type = cfg["action_type"]
        if self.action_type not in ("attitude", "mellinger"):
            raise ValueError(
                f"JaxRacingPolicy only supports action_type='attitude' or "
                f"'mellinger' checkpoints, got {self.action_type!r}"
            )
        if cfg["obs_fn"] != "v3":
            raise ValueError(
                f"JaxRacingPolicy only supports obs_fn='v3' checkpoints (the only "
                f"observation layout ported/verified here), got {cfg['obs_fn']!r}"
            )

        self.action_dim = cfg["action_dim"]
        self.recurrent = bool(cfg.get("recurrent", False))
        if self.action_type == "attitude":
            self.hover_rpm = float(cfg["hover_rpm"])
            self.max_rpm = float(cfg["max_rpm"])
        else:
            # Straight port of crazyflow_interface/__init__.py's own
            # _attitude_cmd bounds — persisted as plain floats by train.py
            # (mellinger_attitude_low/high), not recomputed here, so this
            # file never needs a runtime crazyflow import (see module
            # docstring). Order is [roll, pitch, yaw, thrust], matching the
            # sim's own action convention exactly (NOT the "attitude"
            # action_type's [thrust, roll, pitch, yaw_rate] order above).
            self.attitude_low = np.asarray(cfg["mellinger_attitude_low"], dtype=np.float64)
            self.attitude_high = np.asarray(cfg["mellinger_attitude_high"], dtype=np.float64)
            self.yaw_kp = params.get("yaw_kp", 1.0)

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

        params_path = run_dir / "params.pkl"
        if params_path.exists():
            with open(params_path, "rb") as f:
                import pickle
                self.net_params = pickle.load(f)

            # Warm-up / shape-check the network once at load time so a shape
            # mismatch surfaces immediately at startup, not on the first
            # control cycle in flight.
            dummy_obs = jnp.zeros((21,), dtype=jnp.float32)
            if self.recurrent:
                _, _ = self.network.apply(
                    self.net_params, self.hstate,
                    (dummy_obs[jnp.newaxis, jnp.newaxis, :], jnp.array([[True]])),
                )
            else:
                _ = self.network.apply(self.net_params, dummy_obs)
        else:
            # No params.pkl alongside config_path — fine for onboard-policy
            # mode (only get_observation() is ever called, see class
            # docstring); update() guards against this and raises clearly if
            # actually called without weights, instead of failing on a
            # missing file at construction time.
            self.net_params = None

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

    def get_observation(self, state) -> np.ndarray:
        """Compute the current v3 observation and advance gate-tracking
        state — the first half of update() below, split out so a caller
        that only needs the observation (e.g. logging/diagnostics) doesn't
        have to run the network too.

        state must provide 'x' (world position), 'R' (body->world rotation
        matrix), 'v_b' (body-frame linear velocity), 'w_b' (body-frame
        angular velocity) — all already present in controller_utils.py's
        self.mocap_pose dict."""
        pos = np.asarray(state["x"], dtype=np.float64)
        R = np.asarray(state["R"], dtype=np.float64)
        lin_vel_b = np.asarray(state["v_b"], dtype=np.float64)
        ang_vel_b = np.asarray(state["w_b"], dtype=np.float64)

        self._update_gate_tracking(pos, R)
        return self._build_obs(pos, R, lin_vel_b, ang_vel_b)

    def update(self, state):
        """state must provide 'x' (world position), 'R' (body->world
        rotation matrix), 'v_b' (body-frame linear velocity), 'w_b'
        (body-frame angular velocity) — all already present in
        controller_utils.py's self.mocap_pose dict."""
        if self.net_params is None:
            raise RuntimeError(
                "JaxRacingPolicy.update() called but no params.pkl was found "
                "alongside config.json at construction time. Point config_path "
                "at a run directory that also has params.pkl."
            )
        obs = self.get_observation(state)
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

        if self.action_type == "attitude":
            cmd_thrust = _thrust_action_to_pwm_frac(float(action[0]), self.hover_rpm, self.max_rpm)
            # Fallback-only approximation — see module docstring. Real
            # flight dispatches control_input["raw_action"] via
            # Crazyflie::sendRaceAction to firmware's on-device mixer
            # (crazyflie-firmware/examples/app_race_policy/src/mixer.c)
            # instead of this rate-scale guess.
            roll_dps = float(action[1]) * self.max_roll_rate_dps
            pitch_dps = float(action[2]) * self.max_pitch_rate_dps
            yaw_dps = float(action[3]) * self.max_yaw_rate_dps
            control_input = {
                "cmd_thrust": cmd_thrust,
                "cmd_w": np.array([roll_dps, pitch_dps, yaw_dps]),
                "raw_action": action,
            }
        else:  # mellinger
            R = np.asarray(state["R"], dtype=np.float64)
            mid = (self.attitude_low + self.attitude_high) / 2.0
            half_range = (self.attitude_high - self.attitude_low) / 2.0
            roll_rad, pitch_rad, yaw_rad, thrust_n = mid + action.astype(np.float64) * half_range

            # Real hardware has no genuine 3-axis-angle CRTP setpoint for a
            # pure attitude+thrust command (see the crazyflie_ros plan doc
            # for why) — close the yaw loop with a thin on-host P-wrapper
            # against the current yaw instead, same pattern lsy_drone_racing's
            # own real-hardware AttitudeController uses. Wrapped to the
            # shortest signed angular distance so a yaw crossing +-pi
            # doesn't spike the rate command.
            current_yaw_rad = float(np.arctan2(R[1, 0], R[0, 0]))
            yaw_err_rad = yaw_rad - current_yaw_rad
            yaw_err_rad = np.arctan2(np.sin(yaw_err_rad), np.cos(yaw_err_rad))
            yaw_rate_dps = np.degrees(self.yaw_kp * yaw_err_rad)

            thrust_min_total, thrust_max_total = self.attitude_low[3], self.attitude_high[3]
            cmd_thrust = float(np.clip(
                (thrust_n - thrust_min_total) / (thrust_max_total - thrust_min_total), 0.0, 1.0
            ))
            control_input = {
                "cmd_thrust": cmd_thrust,
                "cmd_attitude": np.array([np.degrees(roll_rad), np.degrees(pitch_rad), yaw_rate_dps]),
            }
        return control_input, obs
