#!/usr/bin/env python3
"""Print joint limits from a URDF file in a copy-pasteable Python dict format."""

import argparse
import xml.etree.ElementTree as ET
import math


def print_joint_limits(urdf_path: str) -> None:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    entries = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        if jtype == "fixed":
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.get("lower", 0))
        upper = float(limit.get("upper", 0))
        lower_deg = round(math.degrees(lower), 1)
        upper_deg = round(math.degrees(upper), 1)
        entries.append((name, lower_deg, upper_deg))

    print("JOINT_LIMITS = {")
    for i, (name, lo, hi) in enumerate(entries):
        comma = "," if i < len(entries) - 1 else ","
        print(f'    "{name}": np.deg2rad([{lo}, {hi}]),')
    print("}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print URDF joint limits as a Python dict.")
    parser.add_argument("urdf", help="Path to the URDF file")
    args = parser.parse_args()
    print_joint_limits(args.urdf)
