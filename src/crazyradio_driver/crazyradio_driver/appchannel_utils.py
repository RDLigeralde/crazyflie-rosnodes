"""Chunking for streaming a race observation over cflib's app-channel.

Mirrors crazyflie_cpp's Crazyflie::sendRaceObservation exactly — same
protocol (1 seq byte + 7 float32 values = 29 bytes/packet, under
firmware's APPCHANNEL_MTU=30), different language. That C++ binding
targets crazyflie_cpp's Crazyflie class, which crazyradio_driver_cpp's
CrazyflieDriver does NOT use (it talks to Connection/Packet directly) —
and crazyradio_driver_cpp isn't even what jirl_bringup's launch file starts
(that's crazyradio_driver, this package, using cflib). This is the
implementation that's actually reachable from the running system.

Firmware side (obs_channel.c) derives each chunk's true valid-float count
from its own compiled-in POLICY_OBS_DIM (policy.h) rather than the wire
carrying it — so the last chunk here is always zero-padded to a full 7
floats even when the observation length isn't a multiple of 7; the padding
is never read as real data on the receiving end.
"""
import struct
from typing import Sequence

FLOATS_PER_CHUNK = 7
# '<' = little-endian, no padding (matches firmware's __attribute__((packed))
# struct exactly): 1 unsigned byte + 7 float32 = 29 bytes.
_STRUCT_FMT = f"<B{FLOATS_PER_CHUNK}f"


def chunk_observation(obs: Sequence[float]) -> list[bytes]:
    """Split obs into app-channel packets, sequence 0..num_chunks-1.

    Caller sends these in order via cflib's appchannel (see
    crazyradio_driver_callbacks.py's race_obs_clbk) — one CRTP packet per
    element of the returned list.
    """
    obs = list(obs)
    n = len(obs)
    num_chunks = (n + FLOATS_PER_CHUNK - 1) // FLOATS_PER_CHUNK
    packets = []
    for chunk in range(num_chunks):
        offset = chunk * FLOATS_PER_CHUNK
        values = obs[offset:offset + FLOATS_PER_CHUNK]
        values = values + [0.0] * (FLOATS_PER_CHUNK - len(values))
        packets.append(struct.pack(_STRUCT_FMT, chunk, *values))
    return packets
