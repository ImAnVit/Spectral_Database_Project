"""
Robust parser for RELAB spectral files.
"""
import re
import numpy as np
from pathlib import Path
from typing import Tuple

NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def parse_relab(filepath: Path) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """Parse RELAB spectrum file robustly.

    RELAB files sometimes contain headers, metadata, and columns. This parser
    extracts the first two numeric columns found per line.

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
                # Skip common header markers
                if line.startswith('#') or line.lower().startswith('header') or line.lower().startswith('!'):
                    continue
                # Try CSV-like parsing first
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
                    # Fallback: find any two numbers in the whole line
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
