"""
Spectrum validation module for quality control.
"""
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


def validate_spectrum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    min_points: int = 50,
    max_missing_fraction: float = 0.20
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate a spectrum for quality and completeness.
    
    Args:
        wavelengths: Array of wavelength values
        reflectance: Array of reflectance values
        min_points: Minimum required valid points
        max_missing_fraction: Maximum allowed fraction of missing values
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {
        'is_valid': True,
        'issues': [],
        'nan_count': 0,
        'negative_count': 0,
        'duplicate_wavelengths': 0,
        'sorted': False,
    }
    
    # Check for empty arrays
    if len(wavelengths) == 0 or len(reflectance) == 0:
        report['is_valid'] = False
        report['issues'].append("Empty spectrum detected")
        return False, report
    
    # Check array length match
    if len(wavelengths) != len(reflectance):
        report['is_valid'] = False
        report['issues'].append(f"Length mismatch: wavelengths={len(wavelengths)}, reflectance={len(reflectance)}")
        return False, report
    
    # Count NaN values
    nan_mask = np.isnan(reflectance)
    report['nan_count'] = int(np.sum(nan_mask))
    nan_fraction = report['nan_count'] / len(reflectance)
    
    if nan_fraction > max_missing_fraction:
        report['is_valid'] = False
        report['issues'].append(f"Too many NaN values: {nan_fraction:.2%} > {max_missing_fraction:.2%}")
    
    # Check for negative reflectance values
    valid_mask = ~nan_mask
    reflectance_valid = reflectance[valid_mask]
    negative_count = np.sum(reflectance_valid < 0)
    report['negative_count'] = int(negative_count)
    
    if negative_count > 0:
        # Negative reflectance values are allowed but flagged; do not reject
        report['issues'].append(f"Negative reflectance values found: {negative_count}")
    
    # Check for duplicate wavelengths
    wavelengths_valid = wavelengths[valid_mask]
    duplicates = len(wavelengths_valid) - len(np.unique(wavelengths_valid))
    report['duplicate_wavelengths'] = duplicates
    
    if duplicates > 0:
        # Flag duplicates but do not reject here; downstream deduplication will handle
        report['issues'].append(f"Duplicate wavelengths found: {duplicates}")
    
    # Check if spectrum has enough valid points
    valid_points = np.sum(valid_mask)
    if valid_points < min_points:
        report['is_valid'] = False
        report['issues'].append(f"Insufficient valid points: {valid_points} < {min_points}")
    
    # Check wavelength ordering
    if len(wavelengths_valid) > 1:
        is_sorted = np.all(np.diff(wavelengths_valid) > 0)
        report['sorted'] = bool(is_sorted)
        if not is_sorted:
            report['issues'].append("Wavelengths not in ascending order")
    
    return report['is_valid'], report


def detect_outliers(
    reflectance: np.ndarray,
    threshold: float = 3.0,
    method: str = 'zscore'
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Detect outliers using statistical methods.
    
    Args:
        reflectance: Array of reflectance values
        threshold: Threshold for outlier detection (standard deviations)
        method: Detection method ('zscore', 'iqr')
        
    Returns:
        Tuple of (outlier_mask, statistics)
    """
    valid_mask = ~np.isnan(reflectance)
    reflectance_valid = reflectance[valid_mask]
    
    stats = {
        'method': method,
        'threshold': threshold,
        'outlier_count': 0,
        'mean': float(np.mean(reflectance_valid)),
        'std': float(np.std(reflectance_valid)),
    }
    
    outlier_mask = np.zeros_like(reflectance, dtype=bool)
    
    if method == 'zscore':
        mean = np.mean(reflectance_valid)
        std = np.std(reflectance_valid)
        
        if std > 0:
            z_scores = np.abs((reflectance_valid - mean) / std)
            outlier_valid = z_scores > threshold
            outlier_mask[np.where(valid_mask)[0]] = outlier_valid
        
        stats['outlier_count'] = int(np.sum(outlier_mask))
    
    elif method == 'iqr':
        q1 = np.percentile(reflectance_valid, 25)
        q3 = np.percentile(reflectance_valid, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_valid = (reflectance_valid < lower_bound) | (reflectance_valid > upper_bound)
        outlier_mask[np.where(valid_mask)[0]] = outlier_valid
        
        stats['outlier_count'] = int(np.sum(outlier_mask))
        stats['q1'] = float(q1)
        stats['q3'] = float(q3)
        stats['iqr'] = float(iqr)
    
    return outlier_mask, stats


def estimate_noise(reflectance: np.ndarray, window_size: int = 5) -> Dict[str, float]:
    """
    Estimate noise level in reflectance spectrum.
    
    Uses local variance estimation within a moving window.
    
    Args:
        reflectance: Array of reflectance values
        window_size: Size of moving window for local variance
        
    Returns:
        Dictionary with noise statistics
    """
    valid_mask = ~np.isnan(reflectance)
    reflectance_valid = reflectance[valid_mask]
    
    if len(reflectance_valid) < window_size:
        return {
            'mean_noise': float(np.std(reflectance_valid)),
            'max_noise': 0.0,
            'noise_estimate_method': 'std'
        }
    
    # Compute local standard deviation
    local_stds = []
    for i in range(len(reflectance_valid) - window_size + 1):
        window = reflectance_valid[i:i + window_size]
        local_stds.append(np.std(window))
    
    local_stds = np.array(local_stds)
    
    return {
        'mean_noise': float(np.mean(local_stds)),
        'max_noise': float(np.max(local_stds)),
        'median_noise': float(np.median(local_stds)),
        'noise_estimate_method': 'local_variance'
    }


def validate_parsed_spectrum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    min_points: int = 10
) -> (bool, Dict):
    """
    Lightweight validation after parsing to ensure arrays are numeric and usable.

    Returns a tuple (is_strictly_valid, report). This function is lenient: it
    flags issues but only rejects immediately for critical failures such as
    missing arrays or entirely non-numeric data. Insufficient points will be
    reported but not necessarily cause immediate rejection (soft validation).
    """

    """
    Lightweight validation after parsing to ensure arrays are numeric and usable.

    Reject if:
    - less than min_points valid pairs
    - any NaNs present
    - wavelengths not sortable or non-numeric
    - reflectance empty or non-numeric

    Returns (is_valid, report)
    """
    report = {'is_valid': True, 'issues': []}

    # Basic presence
    if wavelengths is None or reflectance is None:
        report['is_valid'] = False
        report['issues'].append('Missing arrays')
        return False, report

    try:
        # Convert to numpy arrays
        import numpy as _np
        wl = _np.asarray(wavelengths, dtype=float)
        rf = _np.asarray(reflectance, dtype=float)
    except Exception:
        report['is_valid'] = False
        report['issues'].append('Non-numeric data')
        return False, report

    # Length check (soft)
    if wl.size < min_points or rf.size < min_points:
        report['issues'].append(f'Insufficient points: {wl.size}')
        # do not strictly invalidate here (soft rule)
    
    # NaN check (critical)
    if _np.isnan(wl).any() or _np.isnan(rf).any():
        report['is_valid'] = False
        report['issues'].append('NaN values present')
        return False, report

    # Sortability check
    try:
        diffs = _np.diff(wl)
        # Must have at least monotonic differences (not all zero or NaN)
        if diffs.size == 0:
            report['is_valid'] = False
            report['issues'].append('Wavelength array too short for ordering check')
            return False, report
    except Exception:
        report['is_valid'] = False
        report['issues'].append('Wavelengths not sortable')
        return False, report

    # Reflectance sanity
    if _np.all(_np.isfinite(rf)) is False:
        report['is_valid'] = False
        report['issues'].append('Reflectance contains non-finite values')
        return False, report

    return True, report


def compute_quality_score(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    noise_stats: Dict = None,
    interp_coverage: float = None,
    desired_points: int = 50
) -> Dict:
    """
    Compute a quality score (0-1) for a spectrum based on multiple criteria.

    Returns a dict with 'score' and component scores and warnings.
    """
    import numpy as _np

    result = {
        'score': 0.0,
        'components': {},
        'warnings': []
    }

    try:
        wl = _np.asarray(wavelengths, dtype=float)
        rf = _np.asarray(reflectance, dtype=float)
    except Exception:
        result['warnings'].append('Non-numeric arrays')
        result['score'] = 0.0
        return result

    n_points = wl.size
    # completeness score: ratio of points to desired_points (capped at 1)
    completeness = min(1.0, n_points / float(desired_points)) if desired_points > 0 else 1.0
    result['components']['completeness'] = float(completeness)

    # missing values
    missing = int(_np.isnan(rf).sum())
    missing_frac = missing / n_points if n_points > 0 else 1.0
    completeness *= max(0.0, 1.0 - missing_frac)
    result['components']['missing_fraction'] = float(missing_frac)

    # noise score (lower noise -> higher score)
    if noise_stats and 'mean_noise' in noise_stats:
        mean_noise = float(noise_stats['mean_noise'])
        # assume config max_noise_level roughly maps to 0.05; scale inversely
        noise_score = max(0.0, 1.0 - (mean_noise / (noise_stats.get('scale', mean_noise + 1e-6))))
        result['components']['noise_score'] = float(noise_score)
    else:
        # if unknown, be neutral
        result['components']['noise_score'] = 0.8

    # interpolation coverage
    if interp_coverage is not None:
        coverage_score = float(max(0.0, min(1.0, interp_coverage)))
        result['components']['coverage'] = coverage_score
    else:
        result['components']['coverage'] = 0.8

    # continuity: measure std of first derivative normalized
    try:
        valid_mask = ~_np.isnan(rf)
        rf_valid = rf[valid_mask]
        if rf_valid.size >= 3:
            deriv = _np.diff(rf_valid)
            cont = 1.0 / (1.0 + _np.std(deriv))
            continuity = float(max(0.0, min(1.0, cont)))
        else:
            continuity = 0.5
        result['components']['continuity'] = continuity
    except Exception:
        result['components']['continuity'] = 0.5

    # Combine components with weights
    weights = {
        'completeness': 0.35,
        'noise_score': 0.20,
        'coverage': 0.25,
        'continuity': 0.20
    }

    score = (
        result['components']['completeness'] * weights['completeness'] +
        result['components']['noise_score'] * weights['noise_score'] +
        result['components']['coverage'] * weights['coverage'] +
        result['components']['continuity'] * weights['continuity']
    )

    result['score'] = float(max(0.0, min(1.0, score)))
    result['n_points'] = int(n_points)
    result['wavelength_min'] = float(_np.nanmin(wl)) if n_points > 0 else None
    result['wavelength_max'] = float(_np.nanmax(wl)) if n_points > 0 else None
    result['missing_values'] = int(missing)

    # Warnings for soft rules
    if n_points < desired_points:
        result['warnings'].append(f'Below desired points: {n_points}<{desired_points}')
    if missing_frac > 0.2:
        result['warnings'].append(f'High missing fraction: {missing_frac:.2f}')

    return result
