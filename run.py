#!/usr/bin/env python3

from roop import core

if __name__ == '__main__':
    core.run()
# PATCH for run.py
# Insert these blocks into your existing run.py
# -------------------------
# 1. Add new CLI arguments (place inside argparse parser definitions)

parser.add_argument(
    "--face-texture-strength",
    type=float,
    default=0.35,
    help="Strength of pore restoration (0.0 - 1.0). Default = 0.35"
)

parser.add_argument(
    "--face-micro-sharpen",
    type=float,
    default=0.15,
    help="Fine micro sharpening intensity (0.0 - 1.0). Default = 0.15"
)

parser.add_argument(
    "--face-micro-grain",
    type=float,
    default=4.0,
    help="Micro grain / film grain level. Default = 4.0"
)

# -------------------------
# 2. After args = parser.parse_args(), insert this:

import roop.globals

roop.globals.face_texture_strength = args.face_texture_strength
roop.globals.face_micro_sharpen = args.face_micro_sharpen
roop.globals.face_micro_grain = args.face_micro_grain

# No other modifications required.
# enhancer_skintexture.py will automatically read these values.
