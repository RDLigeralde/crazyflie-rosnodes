from jirl_interfaces.msg import CommandCTBR, OdometryArray
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

def cmd_clbk(self, msg: CommandCTBR):
    if msg.crazyflie_name in self.scf_dict:
        scf = self.scf_dict[msg.crazyflie_name]
        scf.cf.commander.send_setpoint(msg.roll_rate, msg.pitch_rate, -msg.yaw_rate, msg.thrust_pwm)

def mocap_clbk(self, msg: OdometryArray):
    for odom in msg.odom_array:
        cf_name = odom.child_frame_id.split('/')[0]
        if cf_name in self.scf_dict:
            scf = self.scf_dict[cf_name]
            scf.cf.extpos.send_extpose(
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w
            )

# def logger_clbk(self):
#     for log_entry in self.sync_logger.next():
#         print(log_entry)
#         timestamp = log_entry[0]
#         data = log_entry[1]
#         name = log_entry[2]

#         self.get_logger().info('[%d][%s]: %.3s' % (timestamp, name, data))

def reconnect_clbk(self):
    """
    Reconnect callback
    """
    for crazyradio_uri, crazyflie_name in zip(self.crazyradio_uris, self.crazyflie_names):
        if crazyflie_name in self.scf_dict:
            connected = self.scf_dict[crazyflie_name].is_link_open()
            if not connected:
                self.get_logger().error('Crazyflie %s disconnected' % (crazyflie_name))
                self.scf_dict.pop(crazyflie_name)
        else:
            self.get_logger().warn('Trying to connect to Crazyflie %s...' % crazyflie_name)
            try:
                URI = uri_helper.uri_from_env(default=crazyradio_uri)

                self.scf_dict[crazyflie_name] = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
                self.scf_dict[crazyflie_name].open_link()

                self.get_logger().warn('Sending zero command to %s' % crazyradio_uri)
                self.scf_dict[crazyflie_name].cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
                self.get_logger().warn('Connected to %s' % crazyradio_uri)

                self.scf_dict[crazyflie_name].cf.param.set_value('stabilizer.estimator', '2')
                self.scf_dict[crazyflie_name].param.set_value('locSrv.extQuatStdDev', 0.06)
            except Exception as e:
                continue
