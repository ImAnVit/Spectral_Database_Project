"""
Robust parser for RRUFF spectral files.
"""
import re
import numpy as np
from pathlib import Path
from typing import Tuple

NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def parse_rruff(filepath: Path) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """Parse RRUFF spectrum file robustly.

    RRUFF files vary; extract numeric pairs while skipping headers/metadata.

    Returns: (wavelengths, reflectance, valid, error_msg)
    """
    wl = []
    ref = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#') or line.lower().startswith('!'):
                    continue
                # Some RRUFF files have labels: wavelength reflectance
                parts = re.split('[,\s;\t]+', line)
                nums = []
                for p in parts:
                    if NUMBER_RE.match(p):
                        nums.append(p)
                        if len(nums) >= 2:
                            break
                if len(nums) >= 2:
                    try:
                        wl.append(float(nums[0]))
                        ref.append(float(nums[1]))
                    except ValueError:
                        continue
                else:
                    found = NUMBER_RE.findall(line)
                    if len(found) >= 2:
                        try:
                            wl.append(float(found[0]))
                            ref.append(float(found[1]))
                        except ValueError:
                            continue
                    else:
                        continue
        if len(wl) == 0:
            return np.array([]), np.array([]), False, 'No numeric data found'
        return np.array(wl), np.array(ref), True, ''
    except Exception as e:
        return np.array(wl), np.array(ref), False, f'Parsing error: {e}'
