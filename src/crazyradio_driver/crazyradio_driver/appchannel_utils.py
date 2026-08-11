"""Wire packing for streaming one race-policy action over cflib's
app-channel, to crazyflie-firmware's examples/app_race_policy.

Must match action_channel.c's packed ActionPacket struct byte-for-byte:
`uint8_t seq; uint16_t txTickMs; float action[4];` = 1 + 2 + 16 = 19 bytes,
comfortably inside firmware's APPCHANNEL_MTU (30) — one packet per action,
no chunking needed (unlike the old multi-chunk observation-streaming
protocol this replaces, since an action is always exactly 4 floats
regardless of checkpoint).

cf.appchannel.send_packet(bytes)'s exact signature is assumed from cflib's
documented API (send one packet, no built-in multi-packet framing) — cflib
isn't checked out in this repo (git submodule uninitialized) so this
hasn't been verified against the actual pinned version. Confirm once cflib
is available; if the real API differs, only the call site
(crazyradio_driver_callbacks.py's action_clbk) needs to change, not this
packing function.
"""
import struct
from typing import Sequence

# '<' = little-endian, no padding (matches firmware's __attribute__((packed))
# struct exactly): 1 unsigned byte + 1 unsigned short + 4 float32 = 19 bytes.
_STRUCT_FMT = "<BH4f"


def pack_action(seq: int, tx_tick_ms: int, action: Sequence[float]) -> bytes:
    """One ActionPacket-formatted app-channel payload — see module
    docstring for the exact byte layout action_channel.c expects."""
    return struct.pack(_STRUCT_FMT, seq & 0xFF, tx_tick_ms & 0xFFFF, *action)
