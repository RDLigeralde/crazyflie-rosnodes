# Onboard-Mellinger dispatch (JaxRacingPolicy action_type="mellinger") — a
# real attitude setpoint for firmware's stock Mellinger controller
# (stabilizer.controller=2), sent via the legacy RPYT commander
# (Crazyflie::sendSetpoint / cf.commander.send_setpoint). roll_deg/
# pitch_deg are real angles (firmware's default ANGLE stabilization mode),
# NOT rates — deliberately a separate message from CommandCTBR, whose
# roll_rate/pitch_rate fields would silently misdescribe these units if
# reused. yaw_rate_dps is a rate (JaxRacingPolicy's own P-wrapper closes
# the loop against the policy's desired yaw ANGLE on-host — see its
# update() docstring).
string crazyflie_name
float32 roll_deg
float32 pitch_deg
float32 yaw_rate_dps
uint16 thrust_pwm
