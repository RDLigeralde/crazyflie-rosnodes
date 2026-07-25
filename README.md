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
