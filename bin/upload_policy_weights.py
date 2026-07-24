#!/usr/bin/env python3
"""Write a policy_weights.bin blob (mjc_dronetests' export_policy_c.py
--weights-out) into the onboard MEM_TYPE_APP memory over CRTP, swapping the
active race policy without reflashing — see
crazyflie-firmware/examples/app_race_policy/src/policy.h's
policyWeightsWrite()/policy_mem.c for the firmware side of this protocol,
and export_policy_c.py's module docstring for the exact blob layout (magic
+ obsDim/actionDim/numLayers/totalFloats + CRC-32 header, then the raw
float32 payload).

The vehicle must be DISARMED for the firmware to accept the upload at all:
policy.c's armed flag is ctrlRace.obsChanEnable AND policyWeightsValid(), so
an in-progress or already-successful upload while obsChanEnable=1 is
refused outright (see race_controller.c's armed-gate docstring) — use
set_ctrl_race_params.py --disable-onboard first if unsure. A rejected
upload (wrong architecture, bad CRC, or armed) surfaces here as a nonzero
CRTP mem-write status, reported as FAILED below; it never partially/
silently applies (see policyWeightsWrite's own docstring for exactly what
each rejection reason leaves g_weights/weightsValid in).

Usage:
    python3 bin/upload_policy_weights.py --uri radio://0/80/2M/E7E7E701B1 \\
        runs/race/sparse_rpm_seed0/policy_weights.bin

Then verify the upload actually validated before re-arming:
    python3 bin/set_ctrl_race_params.py --uri radio://0/80/2M/E7E7E701B1 --read
(ctrlRace.weightsValid should read 1 — see race_controller.c's LOG_GROUP.)
"""
import argparse
import struct
import sys
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Must match export_policy_c.py's _HEADER_STRUCT/_MAGIC and policy.h's
# PolicyWeightsHeader_t/POLICY_WEIGHTS_MAGIC exactly.
_HEADER_STRUCT = struct.Struct("<6I")  # magic, obsDim, actionDim, numLayers, totalFloats, crc32
_MAGIC = 0x314C4F50
_UPLOAD_TIMEOUT_S = 60.0


def _parse_header(blob: bytes):
    if len(blob) < _HEADER_STRUCT.size:
        raise ValueError(f"blob is only {len(blob)} bytes, shorter than the {_HEADER_STRUCT.size}-byte header")
    magic, obs_dim, action_dim, num_layers, total_floats, crc = _HEADER_STRUCT.unpack(blob[:_HEADER_STRUCT.size])
    if magic != _MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X} (expected 0x{_MAGIC:08X}) — not a policy_weights.bin blob "
                          "produced by export_policy_c.py's export_weights_bin")
    expected_len = _HEADER_STRUCT.size + total_floats * 4
    if len(blob) != expected_len:
        raise ValueError(f"blob length {len(blob)} != header-implied length {expected_len} — truncated/corrupt file")
    return obs_dim, action_dim, num_layers, total_floats, crc


def upload(uri: str, blob_path: str) -> None:
    with open(blob_path, "rb") as f:
        blob = f.read()
    obs_dim, action_dim, num_layers, total_floats, crc = _parse_header(blob)
    print(f"[upload] {blob_path}: obs_dim={obs_dim} action_dim={action_dim} "
          f"num_layers={num_layers} total_floats={total_floats} crc32=0x{crc:08X} "
          f"({len(blob)} bytes)")

    cflib.crtp.init_drivers()
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        refreshed = threading.Event()
        cf.mem.refresh(lambda: refreshed.set())
        if not refreshed.wait(timeout=10):
            print("FAILED: memory refresh (TOC fetch) timed out", file=sys.stderr)
            sys.exit(1)

        mems = cf.mem.get_mems(MemoryElement.TYPE_APP)
        if not mems:
            print("FAILED: no MEM_TYPE_APP memory found on this vehicle — is the "
                  "firmware actually flashed with policy_mem.c's handler registered?",
                  file=sys.stderr)
            sys.exit(1)
        mem = mems[0]
        if mem.size < len(blob):
            print(f"FAILED: onboard memory size {mem.size} < blob size {len(blob)} — "
                  "architecture mismatch (wrong checkpoint exported for this firmware "
                  "build, or policy.h/policy_weights.bin regenerated out of sync).",
                  file=sys.stderr)
            sys.exit(1)

        done = threading.Event()
        failed = threading.Event()

        def _write_done(written_mem, addr):
            if written_mem.id == mem.id:
                done.set()

        def _write_failed(written_mem, addr):
            if written_mem.id == mem.id:
                failed.set()

        cf.mem.mem_write_cb.add_callback(_write_done)
        cf.mem.mem_write_failed_cb.add_callback(_write_failed)

        def _progress(message, percent):
            print(f"\r[upload] {message}: {percent}%", end="", flush=True)

        cf.mem.write(mem, 0, blob, progress_cb=_progress)

        start = time.time()
        while not done.is_set() and not failed.is_set():
            if time.time() - start > _UPLOAD_TIMEOUT_S:
                print(f"\nFAILED: upload timed out after {_UPLOAD_TIMEOUT_S:.0f}s", file=sys.stderr)
                sys.exit(1)
            time.sleep(0.05)
        print()

        if failed.is_set():
            print("FAILED: firmware rejected the upload (armed, out-of-range write, or "
                  "a header that failed to validate against the compiled-in architecture "
                  "— see policyWeightsWrite's docstring). Check ctrlRace.weightsValid/armed "
                  "via set_ctrl_race_params.py --read.", file=sys.stderr)
            sys.exit(1)

        print(f"[upload] Write complete. Verify with:\n"
              f"  python3 bin/set_ctrl_race_params.py --uri {uri} --read")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uri", required=True, help="Crazyradio URI, e.g. radio://0/80/2M/E7E7E701B1")
    p.add_argument("blob", help="Path to a policy_weights.bin produced by export_policy_c.py --weights-out")
    args = p.parse_args()
    upload(args.uri, args.blob)


if __name__ == "__main__":
    main()
