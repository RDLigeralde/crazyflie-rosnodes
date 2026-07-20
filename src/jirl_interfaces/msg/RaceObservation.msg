# Streamed to the Crazyflie's onboard policy over the app-channel (see
# crazyflie-firmware's examples/app_race_policy: obs_channel.c/policy.h).
# obs is race/observations.py's _V3 layout (21 floats), exactly:
#   [lin_vel_b(3), ang_vel_b(3), gravity_b(3), target_b(3),
#    target_b_next(3), normal_b(3), normal_b_next(3)]
# Must match POLICY_OBS_DIM (policy.h) of whichever checkpoint was exported
# to the firmware — a mismatch silently misaligns every chunk boundary on
# the receiving end (see crazyradio_driver's appchannel_utils.chunk_observation).
string crazyflie_name
float32[21] obs
