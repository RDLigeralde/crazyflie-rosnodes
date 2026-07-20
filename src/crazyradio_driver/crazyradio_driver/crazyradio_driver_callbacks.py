from jirl_interfaces.msg import CommandCTBR, OdometryArray, RaceObservation
from jirl_interfaces.srv import Arm
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from .appchannel_utils import chunk_observation

def cmd_clbk(self, msg: CommandCTBR):
    if msg.crazyflie_name in self.scf_dict:
        scf = self.scf_dict[msg.crazyflie_name]
        scf.cf.commander.send_setpoint(msg.roll_rate, msg.pitch_rate, -msg.yaw_rate, msg.thrust_pwm)

def race_obs_clbk(self, msg: RaceObservation):
    """Stream one observation to the onboard policy's app-channel (see
    appchannel_utils.chunk_observation's docstring for the protocol and
    why this — not crazyflie_cpp's Crazyflie::sendRaceObservation — is the
    implementation actually reachable from the running system).

    cf.appchannel.send_packet(bytes)'s exact signature is assumed from
    cflib's documented API (send one packet, no built-in multi-packet
    framing) — cflib isn't checked out in this repo (git submodule
    uninitialized) so this hasn't been verified against the actual pinned
    version. Confirm once cflib is available; if the real API differs,
    only this loop body needs to change, not the chunking protocol itself.
    """
    if msg.crazyflie_name in self.scf_dict:
        scf = self.scf_dict[msg.crazyflie_name]
        for packet in chunk_observation(msg.obs):
            scf.cf.appchannel.send_packet(packet)

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

# Arm callback
def arm_clbk(self, request, response):
    crazyflie_name = request.crazyflie_name
    command = request.command

    if crazyflie_name not in self.scf_dict:
        self.get_logger().error(f'Crazyflie {crazyflie_name} not found')
        response.success = False
        return response
    if command == Arm.Request.ARM:
        self.scf_dict[crazyflie_name].cf.platform.send_arming_request(True)
        self.scf_dict[crazyflie_name].cf.param.set_value('usd.logging', '1')
    elif command == Arm.Request.DISARM:
        self.scf_dict[crazyflie_name].cf.platform.send_arming_request(False)
        self.scf_dict[crazyflie_name].cf.param.set_value('usd.logging', '0')
    else:
        self.get_logger().error(f'Invalid command {command} for Crazyflie {crazyflie_name}')
        response.success = False
        return response
    response.success = True
    self.get_logger().info(f'Crazyflie {crazyflie_name} {"armed" if command == Arm.Request.ARM else "disarmed"}')
    return response