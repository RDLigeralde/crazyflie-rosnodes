import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

figsize = (10, 8)

def set_axes_equal(ax):
  """Set equal scale for all axes."""
  x_limits = ax.get_xlim()
  y_limits = ax.get_ylim()
  z_limits = ax.get_zlim()

  x_range = x_limits[1] - x_limits[0]
  y_range = y_limits[1] - y_limits[0]
  z_range = z_limits[1] - z_limits[0]

  max_range = max(x_range, y_range, z_range) / 2.0

  mid_x = np.mean(x_limits)
  mid_y = np.mean(y_limits)
  mid_z = np.mean(z_limits)

  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

def analyze_ros2_bag(bag_path, t0=0, tf=float('inf')):
  reader = rosbag2_py.SequentialReader()

  storage_id = 'sqlite3' if bag_path.endswith('.db3') else 'mcap'
  storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
  converter_options = rosbag2_py.ConverterOptions()

  reader.open(storage_options, converter_options)

  topic_types = reader.get_all_topics_and_types()
  type_map = {topic.name: topic.type for topic in topic_types}

  print("Available topics:")
  for topic, msg_type in type_map.items():
    print(f"- {topic}: {msg_type}")

  timestamps = []
  gt_pos = {"x": [], "y": [], "z": []}
  gt_quat = {"x": [], "y": [], "z": [], "w": []}
  gt_euler = {"roll": [], "pitch": [], "yaw": []}
  gt_lin_vel = {"x": [], "y": [], "z": []}
  gt_ang_vel = {"x": [], "y": [], "z": []}
  thrust_pwm = []
  roll_rate = []
  pitch_rate = []
  yaw_rate = []

  first_timestamp = None

  while reader.has_next():
    (topic, data, timestamp) = reader.read_next()
    timestamp = timestamp * 1e-9

    if first_timestamp is None:
      first_timestamp = timestamp

    rel_time = timestamp - first_timestamp
    if rel_time < t0 or rel_time > tf:
      continue

    if topic == "/crazy_jirl_01/odom":
      msg_type = type_map[topic]
      msg_class = get_message(msg_type)
      message = deserialize_message(data, msg_class)

      timestamps.append(rel_time)
      gt_pos["x"].append(message.pose.pose.position.x)
      gt_pos["y"].append(message.pose.pose.position.y)
      gt_pos["z"].append(message.pose.pose.position.z)
      gt_quat["x"].append(message.pose.pose.orientation.x)
      gt_quat["y"].append(message.pose.pose.orientation.y)
      gt_quat["z"].append(message.pose.pose.orientation.z)
      gt_quat["w"].append(message.pose.pose.orientation.w)
      gt_lin_vel["x"].append(message.twist.twist.linear.x)
      gt_lin_vel["y"].append(message.twist.twist.linear.y)
      gt_lin_vel["z"].append(message.twist.twist.linear.z)
      gt_ang_vel["x"].append(message.twist.twist.angular.x * 180.0 / np.pi)
      gt_ang_vel["y"].append(message.twist.twist.angular.y * 180.0 / np.pi)
      gt_ang_vel["z"].append(message.twist.twist.angular.z * 180.0 / np.pi)
    elif topic == "/ctbr_cmd":
      msg_type = type_map[topic]
      msg_class = get_message(msg_type)
      message = deserialize_message(data, msg_class)

      thrust_pwm.append(message.thrust_pwm)
      roll_rate.append(message.roll_rate)
      pitch_rate.append(message.pitch_rate)
      yaw_rate.append(message.yaw_rate)

  # Convert quaternion to Euler angles
  quaternions = np.column_stack((gt_quat["x"], gt_quat["y"], gt_quat["z"], gt_quat["w"]))
  euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
  gt_euler["roll"] = euler_angles[:, 0].tolist()
  gt_euler["pitch"] = euler_angles[:, 1].tolist()
  gt_euler["yaw"] = euler_angles[:, 2].tolist()

  # Groun truth data
  fig, axs = plt.subplots(4, 1, figsize=figsize)
  fig.suptitle("Ground truth data")

  axs[0].plot(timestamps, gt_pos["x"], label="$x$")
  axs[0].plot(timestamps, gt_pos["y"], label="$y$")
  axs[0].plot(timestamps, gt_pos["z"], label="$z$")
  axs[0].set_xlabel("Time [s]")
  axs[0].set_ylabel("Positions [m]")
  axs[0].legend()
  axs[0].grid()

  axs[1].plot(timestamps, gt_euler["roll"], label="Roll")
  axs[1].plot(timestamps, gt_euler["pitch"], label="Pitch")
  axs[1].plot(timestamps, gt_euler["yaw"], label="Yaw")
  axs[1].set_xlabel("Time [s]")
  axs[1].set_ylabel("Angles [deg]")
  axs[1].legend()
  axs[1].grid()

  axs[2].plot(timestamps, gt_lin_vel["x"], label="$v_{x}$")
  axs[2].plot(timestamps, gt_lin_vel["y"], label="$v_{y}$")
  axs[2].plot(timestamps, gt_lin_vel["z"], label="$v_{z}$")
  axs[2].set_xlabel("Time [s]")
  axs[2].set_ylabel("Linear velocities [m/s]")
  axs[2].legend()
  axs[2].grid()

  axs[3].plot(timestamps, gt_ang_vel["x"], label=r"$\omega_{x}$")
  axs[3].plot(timestamps, gt_ang_vel["y"], label=r"$\omega_{y}$")
  axs[3].plot(timestamps, gt_ang_vel["z"], label=r"$\omega_{z}$")
  axs[3].set_xlabel("Time [s]")
  axs[3].set_ylabel("Angular velocities [deg/s]")
  axs[3].legend()
  axs[3].grid()

  fig.tight_layout()

  # Rates errors
  fig_rates, axs_rates = plt.subplots(3, 1, figsize=figsize)
  fig_rates.suptitle("Angular velocities comparison")

  axs_rates[0].plot(timestamps, gt_ang_vel["x"], label="Actual")
  axs_rates[0].plot(timestamps, roll_rate, label="Desired")
  axs_rates[0].set_xlabel("Time [s]")
  axs_rates[0].set_ylabel("Roll rate [deg/s]")
  axs_rates[0].legend()
  axs_rates[0].grid()

  axs_rates[1].plot(timestamps, gt_ang_vel["y"], label="Actual")
  axs_rates[1].plot(timestamps, pitch_rate, label="Desired")
  axs_rates[1].set_xlabel("Time [s]")
  axs_rates[1].set_ylabel("Pitch rate [deg/s]")
  axs_rates[1].legend()
  axs_rates[1].grid()

  axs_rates[2].plot(timestamps, gt_ang_vel["z"], label="Actual")
  axs_rates[2].plot(timestamps, yaw_rate, label="Desired")
  axs_rates[2].set_xlabel("Time [s]")
  axs_rates[2].set_ylabel("Yaw rate [deg/s]")
  axs_rates[2].legend()
  axs_rates[2].grid()

  fig_rates.tight_layout()

  # CTBR data
  fig_ctbr, axs_ctbr = plt.subplots(2, 1, figsize=figsize)
  fig_ctbr.suptitle("Commanded CTBR data")

  axs_ctbr[0].plot(timestamps, thrust_pwm, label="Thrust PWM")
  axs_ctbr[0].set_xlabel("Time (s)")
  axs_ctbr[0].set_ylabel("Thrust PWM")
  axs_ctbr[0].grid()

  axs_ctbr[1].plot(timestamps, roll_rate, label="Roll Rate")
  axs_ctbr[1].plot(timestamps, pitch_rate, label="Pitch Rate")
  axs_ctbr[1].plot(timestamps, yaw_rate, label="Yaw Rate")
  axs_ctbr[1].set_xlabel("Time [s]")
  axs_ctbr[1].set_ylabel("Rates [deg/s]")
  axs_ctbr[1].legend()
  axs_ctbr[1].grid()

  # Trajectory 3D plot
  fig3d = plt.figure(figsize=figsize)
  ax3d = fig3d.add_subplot(111, projection='3d')
  ax3d.plot(gt_pos["x"], gt_pos["y"], gt_pos["z"], label="Trajectory")
  ax3d.set_xlabel("x [m]")
  ax3d.set_ylabel("y [m]")
  ax3d.set_zlabel("z [m]")
  ax3d.axis('equal')
  ax3d.set_title("3D Position Trajectory")
  ax3d.legend()

  fig_ctbr.tight_layout()

  # Animation
  def rotate_points(points, roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.apply(points)

  def update(num):
    x, y, z = gt_pos["x"][num], gt_pos["y"][num], gt_pos["z"][num]
    roll, pitch, yaw = gt_euler["roll"][num], gt_euler["pitch"][num], gt_euler["yaw"][num]

    base_points = np.array([
        [-drone_size, -drone_size, 0], [drone_size, drone_size, 0],
        [-drone_size, drone_size, 0], [drone_size, -drone_size, 0]
    ])

    rotated_points = rotate_points(base_points, roll, pitch, yaw)

    drone_x.set_data([x + rotated_points[0, 0], x + rotated_points[1, 0]],
                     [y + rotated_points[0, 1], y + rotated_points[1, 1]])
    drone_x.set_3d_properties([z + rotated_points[0, 2], z + rotated_points[1, 2]])

    drone_y.set_data([x + rotated_points[2, 0], x + rotated_points[3, 0]],
                     [y + rotated_points[2, 1], y + rotated_points[3, 1]])
    drone_y.set_3d_properties([z + rotated_points[2, 2], z + rotated_points[3, 2]])

    trail.set_data(gt_pos["x"][:num+1], gt_pos["y"][:num+1])
    trail.set_3d_properties(gt_pos["z"][:num+1])

    return drone_x, drone_y, trail

  drone_size = 0.1
  anim_fig = plt.figure(figsize=figsize)
  anim_ax = anim_fig.add_subplot(111, projection='3d')
  anim_ax.set_xlim(min(gt_pos["x"]), max(gt_pos["x"]))
  anim_ax.set_ylim(min(gt_pos["y"]), max(gt_pos["y"]))
  anim_ax.set_zlim(0, max(gt_pos["z"]))
  set_axes_equal(anim_ax)
  anim_ax.set_xlabel("x [m]")
  anim_ax.set_ylabel("y [m]")
  anim_ax.set_zlabel("z [m]")
  scatter = anim_ax.scatter([gt_pos["x"][0]], [gt_pos["y"][0]], [gt_pos["z"][0]], color='r')
  trail, = anim_ax.plot([], [], [], 'b--', linewidth=1)
  drone_x, = anim_ax.plot([], [], [], 'r-', linewidth=2)
  drone_y, = anim_ax.plot([], [], [], 'r-', linewidth=2)

  ani = animation.FuncAnimation(anim_fig, update, frames=len(timestamps), interval=50, blit=False)

  # Show all plots
  plt.show()

if __name__ == "__main__":
  import sys
  if len(sys.argv) < 2:
    print("Usage: python analyze_ros2_bag.py <bag_path> [t0] [tf]")
  else:
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    tf = float(sys.argv[3]) if len(sys.argv) > 3 else float('inf')
    analyze_ros2_bag(sys.argv[1], t0, tf)
