from jirl_interfaces.msg import CommandCTBR

def cmd_clbk(self, msg: CommandCTBR):
    scf = self.scf_dict[msg.crazyflie_name]
    scf.cf.commander.send_setpoint(msg.roll_rate, msg.pitch_rate, -msg.yaw_rate, msg.thrust_pwm)  # FIXME