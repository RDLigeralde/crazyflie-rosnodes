# Crazyflie-ROSNodes

Superset of `ros_quad_sim2real` for flying mjc_dronetests RL checkpoints on
real Crazyflie hardware. The policy always runs off-board (this ground
station) — an onboard-network mode existed at one point but has been
removed on both sides (crazyflie-firmware no longer has it either). What
varies is which onboard controller consumes the policy's output:

- **`action_type="attitude"` checkpoints** → a custom open-loop-mixer
  controller running on the vehicle: the ground station streams the
  policy's raw 4-float action over the radio, and firmware runs the exact
  same mixer the sim trained against, bypassing PID entirely.
- **`action_type="mellinger"` checkpoints** → the vehicle's stock Mellinger
  controller: the ground station rescales the policy's action into a real
  physical attitude+thrust setpoint and sends it via the legacy RPYT
  commander.

Both run on the **same** `crazyflie-firmware/examples/app_race_policy`
firmware build — `race_controller.c` arbitrates between PID / the
open-loop mixer / Mellinger / a bench-test mode at runtime, so which path
a given flight uses is a `ctrlRace` param choice, not a reflash. Selected
ground-side by `policy.jax_enable: true` plus whichever `action_type` the
loaded checkpoint's `config.json` declares, and `crazyradio_driver.
mellinger_enable` (which sets `ctrlRace.mellingerEnable`) for the Mellinger
path specifically — see `controller_params.py`/`controller_utils.py`/
`crazyradio_driver_params.py`.

---

## Setup
```bash
git clone --recurse-submodules https://github.com/RDLigeralde/crazyflie-rosnodes
cd crazyflie-rosnodes
```

### Bare Metal
1. ROS2 Installation: [ROS2 Jazzy Installation Docs](https://docs.ros.org/en/jazzy/Installation.html)
2. Shell Configuration: `source /opt/ros/jazzy/setup.bash` (or `.zsh`)
3. Build + Register Nodes: `colcon build && source install/setup.bash`

### Dockerized
1. VSCode Installation: [Download Page](https://code.visualstudio.com/download?_exp_download=fb315fc982)
2. Dev Containers Extension: [Download Page](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Container Startup:
    1. CPU-only: `USER_UID=$(id -u) docker compose -f docker/container-x86-dev/.devcontainer/docker-compose.yaml up -d`
    2. CUDA-enabled (unnecessary if policy not running on workstation): `USER_UID=$(id -u) docker compose -f docker/container-x86-cudev/.devcontainer/docker-compose.yaml up -d`
        - `USER_UID=$(id -u)` ensures equivalent write permissions to the host shell account
4. Attach to Container: open VSCode command pallate with `f1` and run `Dev Containers: Attach to Running Container`
5. Repeat steps 2, 3 from bare metal instructions, replacing `*.bash` with `*.zsh` if needbe

---

## Expected Layout

Three sibling repos under one workspace directory. **Every path below is relative to this layout.**

```
<workspace>/
├── crazyflie_ros/                     ← this repo
│   ├── bin/                           ← set_ctrl_race_params.py
│   └── tools/crazyflie_cpp/           ← Crazyflie::sendRaceAction
├── crazyflie-firmware/
│   └── examples/app_race_policy/      ← firmware for BOTH action_types: build, flash, ctrlRace reference
└── mjx-drone-trainer/                 ← training
    └── runs/<task>/<run-name>/        ← checkpoints (params.pkl + config.json)
```

---

## Flight Sequence

Run everything below from `<workspace>/crazyflie_ros` with the vehicle URI exported:

```bash
export URI=radio://0/80/2M/E7E7E701B1     # your vehicle
```

### 1. Flash the firmware (once — same build for both action_types)

```bash
cd <workspace>/crazyflie-firmware/examples/app_race_policy
make -j$(nproc)
make cload CLOAD_CMDS="-w $URI"
```

See that example's own README for build notes and the full `ctrlRace`
param/log reference. One flash serves both `"attitude"` and `"mellinger"`
checkpoints — `race_controller.c` arbitrates between them (and PID) at
runtime, so switching which checkpoint type you fly next is a `ctrlRace`
param change, not a reflash.

### 2. Arm the mode your checkpoint's `action_type` needs

Check `config.json`'s `action_type` field for the checkpoint you're flying:

- **`"attitude"`** → arm the action stream:
  ```bash
  python3 bin/set_ctrl_race_params.py --uri $URI --enable-stream --read
  ```
  Sets `ctrlRace.actChanEnable=1` — the operator's *intent* to hand over to
  the stream. It does not by itself hand over control; action freshness
  does — see `examples/app_race_policy/README.md`'s **Control Modes**.
- **`"mellinger"`** → set `crazyradio_driver.mellinger_enable: true` in
  `jirl_bringup/config/config.yaml` (step 3 below sets `ctrlRace.
  mellingerEnable=1` at connect time from this) — no manual
  `set_ctrl_race_params.py` call needed for this path.

`mode` (0=PID, 1=stream, 2=bench, 3=mellinger) is a **log** variable, not
listed by `--read` — inspect it in cfclient's log tab or a cflib log
block. Leave whichever param you're not using at its default (0/`false`)
— a fresh action stream always wins over `mellingerEnable` if both
happen to be set (see **Control Modes**), so don't rely on that
precedence instead of just setting the one you mean.

### 3. Bringup (3 terminals)

```bash
ros2 launch jirl_bringup vicon.launch.py
ros2 launch jirl_bringup crazyradio_driver.launch.py
ros2 launch jirl_bringup controller.launch.py namespace:=crazy_jirl_b5
```

Set `policy.jax_enable: true` and `policy.path` (pointing at the
checkpoint's `config.json`) in `jirl_bringup/config/config.yaml` before
bringup — `action_type` itself is read from that `config.json`, not set
separately. For the mellinger path, also set `crazyradio_driver.
mellinger_enable: true` so `ctrlRace.mellingerEnable=1` gets set at connect
time; leave it `false` for the attitude/custom-controller path.

- `policy.path` is **`config.json` itself, not the run directory** —
  `JaxRacingPolicy.__init__` takes a `config_path` and derives `run_dir` as
  its parent. Passing a directory will fail on open.
- `params.pkl` is loaded from alongside `config.json` if present; if not,
  `update()` (network-driven control) raises clearly rather than failing
  on a missing file at startup.

### 4. Fly

```bash
ros2 service call /arm jirl_interfaces/srv/Arm "{crazyflie_name: 'crazy_jirl_b5', command: 0}"
ros2 service call /crazy_jirl_b5/takeoff std_srvs/srv/Trigger
ros2 service call /crazy_jirl_b5/race    std_srvs/srv/Trigger
ros2 service call /crazy_jirl_b5/land    std_srvs/srv/Trigger
```

- `takeoff`/`land` always fly under PID (`ctrlRace.mode=0`) regardless of
  which mode is armed. Only `race` starts streaming actions/attitude
  setpoints for either path.
- Expect `ctrlRace.mode` to read 0 → 1 (attitude path) or 0 → 3 (mellinger
  path) on `race`, and back to 0 on `land`. If it never leaves 0, or the
  vehicle tips over on takeoff, see `examples/app_race_policy/README.md`'s
  **Troubleshooting** — `ctrlRace.mode` plus `ctrlRace.actPackets`
  (attitude path) separate an uplink problem from a policy problem.

---

## Control-loop rate — 48 Hz, PLACEHOLDER

> **This is a stand-in, not a derived value.** It is the number to revisit first if a deployed policy behaves worse on hardware than it did in sim.

The target is **48 Hz**, the control rate every checkpoint was trained at
(`mjx-drone-trainer/configs/race_mjx.yml`: *"1440 ctrl steps @ 30 s"*). The
policy runs once per `single_update()` call, so the mocap callback rate
*is* the policy's effective control rate — matching it to the sim's is the
cheapest way to keep deployed timing close to training.

**Current behavior does not enforce it.** `single_update()` runs once per
incoming Vicon message (`mocap_clbk()`/`multi_mocap_clbk()`), with no timer
and no decimation. The real rate is therefore whatever the mocap system
publishes at (typically 100–200 Hz), not 48 Hz. Closing that gap needs an
explicit decimator or a timer-driven publisher.

Why it matters, in increasing order of severity:

- **Stateless observations (`v3`, `l2f_asymm`)** — mostly harmless. These are pure functions of the current pose, so a faster stream just means more frequent, fresher decisions. For the custom-controller path, rate also affects `ctrlRace.actStaleTicks` headroom (at 48 Hz a packet arrives every 20.8 ms, so the 50 ms default tolerates ~2 missed packets).
- **Action-history observations (`asymm_v2`)** — **this is where it breaks.** The history window is defined in *control steps*, not seconds: 8 steps at the sim's 48 Hz is 167 ms. Run the same policy at 100 Hz and those 8 steps span 80 ms — roughly half the trained window, against a motor-lag time constant of ~0.15 s that the history exists specifically to estimate. The observation silently means something different than it did in training.

Custom-controller path: measuring the actual rate needs no new
instrumentation — read `ctrlRace.actPeriodMs` (the ground's own measured
send period) over a known interval.

Longer-term the honest fix is either decimating to the trained rate, or
retraining at the hardware rate. Pinning the loop to 48 Hz is the interim
option that requires neither.
