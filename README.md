# Crazyflie-ROSNodes

Superset of `ros_quad_sim2real` for sending observations to onboard RL policies instead of ground-computed CTBR commands from ground workstation

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
│   ├── bin/                           ← upload_policy_weights.py, set_ctrl_race_params.py
│   └── tools/crazyflie_cpp/           ← Crazyflie::sendRaceObservation
├── crazyflie-firmware/
│   └── examples/app_race_policy/      ← onboard policy app: build, flash, ctrlRace reference
└── mjx-drone-trainer/                 ← training; also export_policy_c.py
    └── runs/<task>/<run-name>/        ← checkpoints (params.pkl + config.json)
```

---

## Onboard Policy Flight Sequence

Run everything below from `<workspace>/crazyflie_ros` with the vehicle URI exported:

```bash
export URI=radio://0/80/2M/E7E7E701B1     # your vehicle
```

### 1. Build & flash the firmware

See [`crazyflie-firmware/examples/app_race_policy/README.md`](../crazyflie-firmware/examples/app_race_policy/README.md) for the build and `make cload` steps, plus the `ctrlRace` param/log reference and firmware-side troubleshooting.

A fresh flash already has a working policy compiled in, so **step 2 is optional**.

### 2. Weight upload — *skip unless swapping checkpoints*

Only needed to run a **different** checkpoint than the one compiled into the most recent flash.

```bash
python3 ../mjx-drone-trainer/export_policy_c.py \
  --run-dir ../mjx-drone-trainer/runs/race/sparse_attitude_seed0 \
  --out-dir /tmp/policy_export --weights-out /tmp/policy_export/policy_weights.bin
python3 bin/upload_policy_weights.py --uri $URI /tmp/policy_export/policy_weights.bin
```

- Vehicle must be **disarmed**; uploads are refused while armed.
- A bad architecture or failed CRC is rejected whole, never partially applied.
- **A power cycle reverts to the compiled-in checkpoint.** Re-upload after every reboot.
- `export_policy_c.py` needs only `numpy` and `ml_dtypes` — no jax or MuJoCo, so it runs on the flight laptop without the training environment.

### 3. Enable onboard mode (while disarmed)

```bash
python3 bin/set_ctrl_race_params.py --uri $URI --enable-onboard --read
```

- Sets `ctrlRace.obsChanEnable=1` — the operator's *intent* to arm the policy. It does not by itself hand over control; observation freshness does (see the firmware README's **Control Modes**).
- `--read` lists the **param** group only. `weightsValid` and `mode` are **log** variables and will not appear there — inspect them in cfclient's log tab, or with a cflib log block.
- If `weightsValid=0` on a fresh flash, the flashed binary itself is suspect — recheck the build before flying.

### 4. Bringup (3 terminals)

```bash
ros2 launch jirl_bringup vicon.launch.py
ros2 launch jirl_bringup crazyradio_driver.launch.py
ros2 launch jirl_bringup controller.launch.py \
  namespace:=crazy_jirl_b5 \
  onboard_policy_enable:=true \
  onboard_policy_checkpoint_path:=../mjx-drone-trainer/runs/race/sparse_attitude_seed0/config.json
```

- `onboard_policy_checkpoint_path` points at **`config.json` itself, not the run directory** — `JaxRacingPolicy.__init__` takes a `config_path` and derives `run_dir` as its parent. Passing a directory will fail on open.
- It resolves against the **launch process's working directory**, not the package — the value above assumes you launched from `crazyflie_ros/`. Use an absolute path otherwise.
- It is **ground-side only**: the node needs `config.json` for gate geometry and observation layout, nothing else. `params.pkl` is optional — if it isn't sitting alongside `config.json`, it is simply never loaded, so a flight laptop only has to carry the config. The weights that actually fly live on the Crazyflie.
- Nothing cross-checks the two: make sure this checkpoint matches whatever is really running onboard (the flashed one, or step 2's upload).

### 5. Fly

```bash
ros2 service call /arm jirl_interfaces/srv/Arm "{crazyflie_name: 'crazy_jirl_b5', command: 0}"
ros2 service call /crazy_jirl_b5/takeoff std_srvs/srv/Trigger
ros2 service call /crazy_jirl_b5/race    std_srvs/srv/Trigger
ros2 service call /crazy_jirl_b5/land    std_srvs/srv/Trigger
```

- `takeoff`/`land` fly under the onboard **PID** (`ctrlRace.mode=0`). Only `race` starts the `/race_obs` stream that promotes the vehicle to `mode=1`.
- Expect `ctrlRace.mode` to read 0 → 1 on `race`, and 1 → 0 on `land`.
- If it never reaches 1, or tips over on takeoff, see the firmware README's **Troubleshooting** — `ctrlRace.mode` and `ctrlRace.obsFrames` separate an uplink problem from a policy problem.

---

## Observation Stream Rate — 48 Hz, PLACEHOLDER

> **This is a stand-in, not a derived value.** It is the number to revisit first if an onboard policy behaves worse on hardware than it did in sim.

The target is **48 Hz**, chosen only because it is the control rate every checkpoint was trained at (`mjx-drone-trainer/configs/race_mjx.yml`: *"1440 ctrl steps @ 30 s"*). The onboard policy runs once per completed observation frame, so the `/race_obs` publish rate *is* the policy's control rate — matching it to the sim's is the cheapest way to keep deployed timing close to training.

**Current behavior does not enforce it.** `/race_obs` is published from `single_update()` inside `mocap_clbk()` — once per incoming Vicon message, with no timer and no decimation. The real rate is therefore whatever the mocap system publishes at (typically 100–200 Hz), not 48 Hz. Closing that gap needs an explicit decimator or a timer-driven publisher.

Why it matters, in increasing order of severity:

- **Stateless observations (`v3`, `l2f_asymm`)** — mostly harmless. These are pure functions of the current pose, so a faster stream just means more frequent, fresher decisions. Rate affects `ctrlRace.obsStaleTicks` headroom (at 48 Hz a frame arrives every 20.8 ms, so the 50 ms default tolerates ~2 missed frames) and onboard inference cost, nothing else.
- **Action-history observations (`asymm_v2`)** — **this is where it breaks.** The history window is defined in *control steps*, not seconds: 8 steps at the sim's 48 Hz is 167 ms. Run the same policy at 100 Hz and those 8 steps span 80 ms — roughly half the trained window, against a motor-lag time constant of ~0.15 s that the history exists specifically to estimate. The observation silently means something different than it did in training.

Measuring the actual rate needs no new instrumentation: read `ctrlRace.obsFrames` (cumulative completed frames) over a known interval.

Longer-term the honest fix is either decimating onboard inference to the trained rate, or retraining at the hardware rate. Pinning the stream to 48 Hz is the interim option that requires neither.
