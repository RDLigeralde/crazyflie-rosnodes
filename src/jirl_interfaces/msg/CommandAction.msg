# Custom controller / open-loop mixer dispatch (JaxRacingPolicy
# action_type="attitude") — streamed to crazyflie-firmware's
# examples/app_race_policy over the app-channel (see
# Crazyflie::sendRaceAction, the transmitting half, and action_channel.c,
# the receiving half). action is [thrust_norm, roll, pitch, yaw_rate],
# each in [-1, 1] — the exact convention crazyflow_interface/_mixer.py's
# mix_attitude_rpm (ported byte-exact to examples/app_race_policy/src/
# mixer.c) expects. seq/tx_tick_ms let the firmware reject stale/reordered
# packets and measure the true send period — see action_channel.c's
# ActionPacket struct, which this message must stay byte-compatible with.
string crazyflie_name
uint8 seq
uint16 tx_tick_ms
float32[4] action
