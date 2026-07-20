#!/usr/bin/env python3
"""Set/read examples/app_race_policy's ctrlRace param group over the radio
link, standalone from the ROS2 stack (no controller/crazyradio_driver nodes
need to be running — this talks to the Crazyflie directly via cflib, the
same library crazyradio_driver itself uses).

Exists because nothing currently writes these automatically: obsChanEnable
defaults to 0 on every boot (see race_controller.c's file docstring), and
hoverRpm/maxRpm/kf/differentialFrac ship with mjc_dronetests' sim-nominal
defaults baked in at compile time (matching whichever checkpoint was last
exported — see export_policy_c.py) rather than a real vehicle's calibrated
values. This script is the manual bridge until/unless that gets wrapped in
a ROS service.

Usage
-----
Enable onboard-policy mode (the common case — do this once per session,
before arming, per the flight-sequence runbook):
    python3 bin/set_ctrl_race_params.py --uri radio://0/80/2M/E7E7E701B1 --enable-onboard

Disable it (fall back to the ctrlRace group's own action0..3 bench-test
params):
    python3 bin/set_ctrl_race_params.py --uri radio://0/80/2M/E7E7E701B1 --disable-onboard

Push real per-vehicle calibration values (system-ID'd, not sim-nominal —
see KNOWN_PARAMS below for every name race_controller.c actually exposes;
there's no motorTau here, real motor lag isn't a firmware-side concept, it
only exists as a sim-side domain-randomization parameter):
    python3 bin/set_ctrl_race_params.py --uri radio://0/80/2M/E7E7E701B1 \\
        --set hoverRpm=15217.0 --set maxRpm=27693.0 --set differentialFrac=0.02

Read back the whole group without changing anything:
    python3 bin/set_ctrl_race_params.py --uri radio://0/80/2M/E7E7E701B1 --read
"""
import argparse
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

GROUP = "ctrlRace"

# Every param race_controller.c's PARAM_GROUP_START(ctrlRace) currently
# declares — kept here only for --read's default listing; --set works for
# any name regardless (e.g. a future param this list hasn't been updated
# for yet), it's not a validating whitelist.
KNOWN_PARAMS = [
    "obsChanEnable",
    "hoverRpm",
    "maxRpm",
    "kf",
    "differentialFrac",
    "action0",
    "action1",
    "action2",
    "action3",
]


def parse_set_args(set_args):
    """'name=value' strings -> [(name, value_str), ...]. Value stays a
    string; cflib's Param.set_value coerces it (float(value) or int(value))
    based on the param's actual TOC-reported type — see its own source
    (cflib/crazyflie/param.py) for exactly how, no need to duplicate that
    logic here."""
    parsed = []
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        parsed.append((name.strip(), value.strip()))
    return parsed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uri", required=True, help="Crazyradio URI, e.g. radio://0/80/2M/E7E7E701B1")
    p.add_argument("--enable-onboard", action="store_true", help="Shortcut for --set obsChanEnable=1")
    p.add_argument("--disable-onboard", action="store_true", help="Shortcut for --set obsChanEnable=0")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Set ctrlRace.NAME to VALUE. Repeatable.")
    p.add_argument("--read", action="store_true",
                    help="Print current values of every param in KNOWN_PARAMS after applying any --set/--enable/--disable above.")
    args = p.parse_args()

    if args.enable_onboard and args.disable_onboard:
        p.error("--enable-onboard and --disable-onboard are mutually exclusive")

    sets = parse_set_args(args.set)
    if args.enable_onboard:
        sets.append(("obsChanEnable", "1"))
    if args.disable_onboard:
        sets.append(("obsChanEnable", "0"))

    if not sets and not args.read:
        p.error("nothing to do — pass --set/--enable-onboard/--disable-onboard and/or --read")

    cflib.crtp.init_drivers()
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        param = scf.cf.param

        for name, value in sets:
            complete_name = f"{GROUP}.{name}"
            try:
                param.set_value(complete_name, value)
                print(f"set {complete_name} = {value}")
            except KeyError:
                print(f"FAILED: {complete_name} not in param TOC — "
                      f"is the firmware actually flashed with this param group?", file=sys.stderr)
                sys.exit(1)
            except AttributeError as e:
                print(f"FAILED: {complete_name}: {e}", file=sys.stderr)
                sys.exit(1)

        if args.read:
            print(f"--- {GROUP} (post-write values) ---")
            for name in KNOWN_PARAMS:
                complete_name = f"{GROUP}.{name}"
                try:
                    value = param.get_value(complete_name)
                    print(f"{complete_name} = {value}")
                except KeyError:
                    print(f"{complete_name} = <not in TOC>")


if __name__ == "__main__":
    main()
