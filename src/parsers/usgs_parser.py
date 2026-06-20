"""
Robust USGS spectral parser that handles:
 - two-column files (wavelength, reflectance)
 - one-column files (reflectance only)
 - variable-length headers and mixed formatting

Behavior:
 - Reads file line-by-line (utf-8, errors='replace')
 - Detects the first sustained numeric block and infers column structure
 - If wavelengths are missing, reconstructs standard grid (400..2500 step 1)
 - Stops collection after a sustained run of non-numeric lines
 - Returns (wavelengths, reflectance, valid, error_message)
"""
from pathlib import Path
from typing import Tuple, Optional, List
import re
import numpy as np

_NUMBER_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

DEFAULT_WL_MIN = 400.0
DEFAULT_WL_MAX = 2500.0
DEFAULT_WL_STEP = 1.0
MIN_POINTS = 10
CONSECUTIVE_INVALID_THRESHOLD = 5
MIN_CONSECUTIVE_NUMERIC = 3  # for block detection fallback


def _extract_numbers(line: str) -> List[str]:
    return _NUMBER_RE.findall(line)


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _make_standard_grid(wl_min=DEFAULT_WL_MIN, wl_max=DEFAULT_WL_MAX, step=DEFAULT_WL_STEP) -> np.ndarray:
    return np.arange(wl_min, wl_max + 1e-12, step, dtype=float)


def parse_usgs_file(path: Path,
                    wl_min: float = DEFAULT_WL_MIN,
                    wl_max: float = DEFAULT_WL_MAX,
                    wl_step: float = DEFAULT_WL_STEP,
                    min_points: int = MIN_POINTS,
                    consecutive_invalid_threshold: int = CONSECUTIVE_INVALID_THRESHOLD
                    ) -> Tuple[np.ndarray, np.ndarray, bool, Optional[str]]:
    """
    Parse a USGS text file robustly and return (wavelengths, reflectance, valid, error).

    - If file contains two numeric columns, parse as (wl, ref).
    - If file contains single numeric column, treat as reflectance-only and reconstruct wavelengths
      using the standard grid (wl_min..wl_max step wl_step). Reflectance length is truncated or
      aligned to the grid length (reflectance -> grid[:N]).
    - Reject only if < min_points or no numeric data found.
    """
    if not isinstance(path, Path):
        path = Path(path)

    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return np.array([], dtype=float), np.array([], dtype=float), False, f"io_error: {e}"

    lines = text.splitlines()

    # Find the longest sustained numeric block (lines with at least one numeric token)
    best_start = best_end = None
    cur_start = cur_count = 0
    in_block = False
    numeric_counts = []

    for i, raw in enumerate(lines):
        line = raw.strip()
        nums = _extract_numbers(line)
        count = len(nums)
        numeric_counts.append(count)
        if count >= 1:
            if not in_block:
                in_block = True
                cur_start = i
                cur_count = 1
            else:
                cur_count += 1
        else:
            if in_block:
                if best_start is None or cur_count > (best_end - best_start + 1 if best_end is not None else 0):
                    best_start = cur_start
                    best_end = i - 1
                in_block = False
                cur_count = 0
    if in_block:
        if best_start is None or cur_count > (best_end - best_start + 1 if best_end is not None else 0):
            best_start = cur_start
            best_end = len(lines) - 1

    # fallback: find any short consecutive numeric window
    if best_start is None:
        consec = 0
        start_idx = 0
        for i, c in enumerate(numeric_counts):
            if c >= 1:
                if consec == 0:
                    start_idx = i
                consec += 1
                if consec >= MIN_CONSECUTIVE_NUMERIC:
                    best_start = start_idx
                    best_end = i
                    break
            else:
                consec = 0

    if best_start is None:
        return np.array([], dtype=float), np.array([], dtype=float), False, "no_numeric_data_found"

    # extend block forward tolerating occasional non-numeric lines
    start = best_start
    end = best_end
    i = end + 1
    consec_invalid = 0
    while i < len(lines):
        c = len(_extract_numbers(lines[i].strip()))
        if c >= 1:
            end = i
            consec_invalid = 0
        else:
            consec_invalid += 1
            if consec_invalid >= consecutive_invalid_threshold:
                break
        i += 1

    # inspect counts in block
    counts = [len(_extract_numbers(lines[j].strip())) for j in range(start, end + 1)]
    if not counts:
        return np.array([], dtype=float), np.array([], dtype=float), False, "no_numeric_data_in_block"

    n_lines = len(counts)
    n_ge2 = sum(1 for c in counts if c >= 2)
    n_eq1 = sum(1 for c in counts if c == 1)

    two_column = n_ge2 >= max(1, int(0.6 * n_lines))
    one_column = n_eq1 >= max(1, int(0.6 * n_lines))

    wls = []
    refs = []

    if two_column:
        for j in range(start, end + 1):
            nums = _extract_numbers(lines[j].strip())
            if len(nums) < 2:
                continue
            wl = _to_float(nums[0])
            rf = _to_float(nums[1])
            if wl is None or rf is None:
                continue
            wls.append(wl)
            refs.append(rf)
    elif one_column:
        for j in range(start, end + 1):
            nums = _extract_numbers(lines[j].strip())
            if len(nums) < 1:
                continue
            rf = _to_float(nums[0])
            if rf is None:
                continue
            refs.append(rf)
    else:
        for j in range(start, end + 1):
            nums = _extract_numbers(lines[j].strip())
            if len(nums) >= 2:
                wl = _to_float(nums[0])
                rf = _to_float(nums[1])
                if wl is None or rf is None:
                    continue
                wls.append(wl)
                refs.append(rf)
            elif len(nums) == 1:
                rf = _to_float(nums[0])
                if rf is None:
                    continue
                refs.append(rf)

    try:
        if len(wls) >= min(min_points, MIN_POINTS) and len(wls) == len(refs):
            wl_arr = np.array(wls, dtype=float)
            rf_arr = np.array(refs, dtype=float)
        elif len(refs) >= min(min_points, MIN_POINTS) and len(wls) == 0:
            rf_arr = np.array(refs, dtype=float)
            grid = _make_standard_grid(wl_min, wl_max, wl_step)
            if rf_arr.size >= grid.size:
                rf_arr = rf_arr[:grid.size]
                wl_arr = grid
            else:
                wl_arr = grid[: rf_arr.size]
        else:
            if len(wls) > 0 and len(wls) == len(refs):
                wl_arr = np.array(wls, dtype=float)
                rf_arr = np.array(refs, dtype=float)
                return wl_arr, rf_arr, False, f"Insufficient points: {wl_arr.size}"
            elif len(refs) > 0:
                rf_arr = np.array(refs, dtype=float)
                return np.array([], dtype=float), rf_arr, False, f"Insufficient points: {rf_arr.size}"
            else:
                return np.array([], dtype=float), np.array([], dtype=float), False, "no_parsed_points"

        mask = np.isfinite(wl_arr) & np.isfinite(rf_arr)
        wl_arr = wl_arr[mask]
        rf_arr = rf_arr[mask]

        if wl_arr.size < min_points:
            return wl_arr, rf_arr, False, f"Insufficient points: {wl_arr.size}"

        rounded = np.round(wl_arr, 6)
        uniq = {}
        for rw, r in zip(rounded, rf_arr):
            uniq.setdefault(rw, []).append(r)
        wl_sorted = np.array(sorted(uniq.keys()), dtype=float)
        rf_agg = np.array([float(np.mean(uniq[w])) for w in wl_sorted], dtype=float)

        if wl_sorted.size >= 2 and wl_sorted[0] > wl_sorted[-1]:
            wl_sorted = wl_sorted[::-1]
            rf_agg = rf_agg[::-1]

        if wl_sorted.size < min_points:
            return wl_sorted, rf_agg, False, f"Insufficient points: {wl_sorted.size}"

        return wl_sorted, rf_agg, True, None

    except Exception as e:
        return np.array([], dtype=float), np.array([], dtype=float), False, f"postprocess_error: {e}"
