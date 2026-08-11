from jirl_interfaces.msg import CommandCTBR, CommandAction, CommandAttitude, OdometryArray
from jirl_interfaces.srv import Arm
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from .appchannel_utils import pack_action

def cmd_clbk(self, msg: CommandCTBR):
    if msg.crazyflie_name in self.scf_dict:
        scf = self.scf_dict[msg.crazyflie_name]
        scf.cf.commander.send_setpoint(msg.roll_rate, msg.pitch_rate, -msg.yaw_rate, msg.thrust_pwm)

def action_clbk(self, msg: CommandAction):
    """Custom controller / open-loop mixer path (JaxRacingPolicy
    action_type="attitude") — stream one action packet to
    crazyflie-firmware's examples/app_race_policy over the app-channel
    (action_channel.c is the receiving half; see pack_action's docstring
    for the wire format both sides must agree on byte-for-byte).

    cf.appchannel.send_packet(bytes)'s exact signature is assumed from
    cflib's documented API (send one packet, no built-in multi-packet
    framing) — cflib isn't checked out in this repo (git submodule
    uninitialized) so this hasn't been verified against the actual pinned
    version. Confirm once cflib is available.
    """
    if msg.crazyflie_name in self.scf_dict:
        scf = self.scf_dict[msg.crazyflie_name]
        scf.cf.appchannel.send_packet(pack_action(msg.seq, msg.tx_tick_ms, msg.action))

def attitude_clbk(self, msg: CommandAttitude):
    """Onboard-Mellinger path (JaxRacingPolicy action_type="mellinger") —
    a real attitude setpoint via the legacy RPYT commander, for firmware's
    stock Mellinger controller (stabilizer.controller=2, set at connect
    time — see reconnect_clbk) to consume. Reuses the exact same
    send_setpoint call cmd_clbk uses for the (rate-based) CTBR path — the
    difference is entirely in what the caller means by roll/pitch (angles,
    not rates) and in stabilizer.controller's value, not in the wire call
    itself.
    """
    if msg.crazyflie_name in self.scf_dict:
        scf = self.scf_dict[msg.crazyflie_name]
        scf.cf.commander.send_setpoint(msg.roll_deg, msg.pitch_deg, -msg.yaw_rate_dps, msg.thrust_pwm)

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
                if self.mellinger_enable:
                    # Onboard-Mellinger path — sets the operator's INTENT to
                    # route the standard setpoint through
                    # controllerMellingerFirmware() instead of
                    # controllerPid() (see crazyflie-firmware's
                    # examples/app_race_policy/src/race_controller.c). One
                    # firmware build now serves both this path and the
                    # attitude-mixer path — NOT stabilizer.controller (that
                    # param switches the ACTIVE controller away from
                    # controllerOutOfTree() entirely, discarding
                    # race_controller.c's own arbitration/PID-fallback along
                    # with it; see crazyradio_driver_params.py).
                    self.scf_dict[crazyflie_name].cf.param.set_value('ctrlRace.mellingerEnable', '1')
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