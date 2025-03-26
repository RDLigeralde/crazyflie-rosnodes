from jirl_interfaces.msg import CommandCTBR

def cmd_clbk(self, msg: CommandCTBR):
    scf = self.scf_dict[msg.crazyflie_name]
    scf.cf.commander.send_setpoint(msg.roll_rate, msg.pitch_rate, -msg.yaw_rate, msg.thrust_pwm)  # FIXME

# def logger_clbk(self):
#     for log_entry in self.sync_logger.next():
#         print(log_entry)
#         timestamp = log_entry[0]
#         data = log_entry[1]
#         name = log_entry[2]

#         self.get_logger().info('[%d][%s]: %.3s' % (timestamp, name, data))