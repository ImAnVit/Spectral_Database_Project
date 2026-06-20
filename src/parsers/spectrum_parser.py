"""
Unified spectrum parser facade.

Responsibilities:
 - Heuristically route files to library-specific parsers (usgs/relab/rruff) by path.
 - Provide a robust generic fallback parser for mixed/unlabelled files.
 - Return standardized tuple:
     (wavelengths: np.ndarray,
      reflectance: np.ndarray,
      format_type: str,   # e.g. 'usgs-2col', 'usgs-1col', 'relab', 'rruff', 'generic-2col', 'generic-1col'
      valid: bool,
      error: Optional[str])
"""
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

# import library-specific parser (usgs). RELAB/RRUFF parsers remain untouched.
from . import usgs_parser

# Try to import relab/rruff parsers if present; if not, fallback to generic
try:
    from .relab_parser import parse_relab as parse_relab_file  # type: ignore
except Exception:
    parse_relab_file = None  # type: ignore

try:
    from .rruff_parser import parse_rruff as parse_rruff_file  # type: ignore
except Exception:
    parse_rruff_file = None  # type: ignore


def _heuristic_library_from_path(path: Path) -> str:
    s = str(path).lower()
    if 'usgs' in s:
        return 'usgs'
    if 'relab' in s:
        return 'relab'
    if 'rruff' in s:
        return 'rruff'
    # fallback: unknown
    return 'unknown'


def parse_spectrum(path: Path,
                   wl_min: float = 400.0,
                   wl_max: float = 2500.0,
                   wl_step: float = 1.0,
                   min_points: int = 10) -> Tuple[np.ndarray, np.ndarray, str, bool, Optional[str]]:
    """
    Unified parser entrypoint.

    Returns:
      wavelengths, reflectance, format_type, valid, error
    """
    if not isinstance(path, Path):
        path = Path(path)

    lib_guess = _heuristic_library_from_path(path)

    # Prefer library-specific parser when available
    if lib_guess == 'usgs':
        wl, rf, valid, err = usgs_parser.parse_usgs_file(path, min_points=min_points)
        fmt = 'usgs-2col' if valid and wl.size > 0 and rf.size > 0 and wl.size == rf.size else ('usgs-1col' if valid and rf.size > 0 and wl.size == 0 else 'usgs-unknown')
        return wl, rf, fmt, valid, err

    if lib_guess == 'relab' and parse_relab_file is not None:
        try:
            wl, rf, valid, err = parse_relab_file(path)
            fmt = 'relab'
            return wl, rf, fmt, valid, err
        except Exception as e:
            # fallback to generic
            pass

    if lib_guess == 'rruff' and parse_rruff_file is not None:
        try:
            wl, rf, valid, err = parse_rruff_file(path)
            fmt = 'rruff'
            return wl, rf, fmt, valid, err
        except Exception:
            pass

    # Generic fallback using the USGS parser's detection logic (handles 1-col and 2-col)
    wl, rf, valid, err = usgs_parser.parse_usgs_file(path, wl_min=wl_min, wl_max=wl_max, wl_step=wl_step, min_points=min_points)
    fmt = 'generic-2col' if valid and wl.size > 0 and rf.size > 0 and wl.size == rf.size else ('generic-1col' if valid and rf.size > 0 and wl.size == 0 else 'generic-unknown')
    return wl, rf, fmt, valid, err
