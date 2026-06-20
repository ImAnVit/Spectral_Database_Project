"""
Spectrum cleaning and preprocessing module with robust parsing wrapper.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

from .validate_spectra import validate_spectrum, detect_outliers, estimate_noise
from .spectral_similarity import detect_spectral_duplicates
from .spectral_distortion import compute_spectral_distortion_score, compute_variance_preservation_ratio, flag_significantly_altered
from src.parsers import get_parser_for_path


logger = logging.getLogger(__name__)


def read_spectrum(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Robust wrapper to parse spectrum using library-specific parsers.

    Returns:
        wavelengths, reflectance, metadata
    """
    metadata = {'source_file': filepath.name, 'parse_error': ''}
    try:
        parser = get_parser_for_path(str(filepath))
        if parser is None:
            # Fallback: attempt a generic CSV read but robust
            try:
                df = pd.read_csv(filepath, comment='#', header=None, dtype=str, encoding='utf-8', error_bad_lines=False)
            except Exception:
                # Last resort: read lines and extract numbers
                from src.parsers import parse_usgs
                wl, ref, valid, err = parse_usgs(filepath)
                if not valid:
                    metadata['parse_error'] = err
                return wl, ref, metadata
            # Find first two numeric columns
            numeric_cols = []
            for col in df.columns:
                try:
                    col_vals = pd.to_numeric(df[col], errors='coerce')
                    if col_vals.notna().sum() > 0:
                        numeric_cols.append(col)
                    if len(numeric_cols) >= 2:
                        break
                except Exception:
                    continue
            if len(numeric_cols) < 2:
                metadata['parse_error'] = 'Could not find two numeric columns'
                return np.array([]), np.array([]), metadata
            try:
                wavelengths = pd.to_numeric(df[numeric_cols[0]], errors='coerce').to_numpy(dtype=float)
                reflectance = pd.to_numeric(df[numeric_cols[1]], errors='coerce').to_numpy(dtype=float)
                return wavelengths, reflectance, metadata
            except Exception as e:
                metadata['parse_error'] = f'CSV fallback parse error: {e}'
                return np.array([]), np.array([]), metadata
        else:
            wl, ref, valid, err = parser(filepath)
            if not valid:
                metadata['parse_error'] = err
            return wl, ref, metadata
    except Exception as e:
        metadata['parse_error'] = f'Unhandled parse exception: {e}'
        logger.debug(metadata['parse_error'])
        return np.array([]), np.array([]), metadata


def remove_nans(wavelengths: np.ndarray, reflectance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove NaN values from spectrum."""
    valid_mask = ~(np.isnan(wavelengths) | np.isnan(reflectance))
    nan_removed = int(np.sum(~valid_mask))
    return wavelengths[valid_mask], reflectance[valid_mask], nan_removed


def remove_duplicates(wavelengths: np.ndarray, reflectance: np.ndarray, use_spectral_similarity: bool = True) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """Remove duplicate wavelengths using spectral similarity-based detection.
    
    Args:
        wavelengths: Wavelength array
        reflectance: Reflectance array
        use_spectral_similarity: If True, use spectral similarity; if False, use exact matching
        
    Returns:
        Tuple of (wavelengths_unique, reflectance_unique, duplicates_removed, stats)
    """
    if use_spectral_similarity:
        wl_unique, ref_unique, dups_removed, stats = detect_spectral_duplicates(
            wavelengths, reflectance, similarity_threshold=0.995, method='combined'
        )
        return wl_unique, ref_unique, dups_removed, stats
    else:
        # Fallback to exact matching
        unique_indices = np.unique(wavelengths, return_index=True)[1]
        unique_indices = np.sort(unique_indices)
        duplicates_removed = int(len(wavelengths) - len(unique_indices))
        return wavelengths[unique_indices], reflectance[unique_indices], duplicates_removed, {'method': 'exact'}


def remove_negative_values(wavelengths: np.ndarray, reflectance: np.ndarray, clip_to_zero: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """Handle negative reflectance values.

    By default, clip negative reflectance to zero instead of removing points.
    Returns the (possibly modified) arrays and number of values clipped.
    """
    neg_mask = reflectance < 0
    negatives_count = int(np.sum(neg_mask))
    if negatives_count == 0:
        return wavelengths, reflectance, 0
    if clip_to_zero:
        reflectance = reflectance.copy()
        reflectance[neg_mask] = 0.0
        return wavelengths, reflectance, negatives_count
    else:
        # fallback: remove negative points (legacy behavior)
        valid_mask = reflectance >= 0
        negatives_removed = int(np.sum(~valid_mask))
        return wavelengths[valid_mask], reflectance[valid_mask], negatives_removed


def sort_by_wavelength(wavelengths: np.ndarray, reflectance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort spectrum by wavelength in ascending order."""
    sort_indices = np.argsort(wavelengths)
    return wavelengths[sort_indices], reflectance[sort_indices]


def apply_smoothing(
    reflectance: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    library: str = 'default'
) -> Tuple[np.ndarray, bool]:
    """
    Apply Savitzky-Golay smoothing filter with library-specific intensity.
    
    Library-specific smoothing:
    - RELAB: minimal smoothing (preserve high-quality lab data)
    - RRUFF: moderate smoothing
    - USGS: moderate smoothing
    """
    # Library-specific window sizes (smaller = less aggressive)
    library_windows = {
        'relab': 7,   # Minimal smoothing for high-quality data
        'rruff': 11,  # Moderate
        'usgs': 11,   # Moderate
        'default': 11
    }
    
    window_length = library_windows.get(library.lower(), window_length)
    
    try:
        if window_length % 2 == 0:
            window_length -= 1
        if len(reflectance) < window_length:
            logger.warning(f"Not enough points ({len(reflectance)}) for smoothing with window {window_length}")
            return reflectance, False
        smoothed = savgol_filter(reflectance, window_length, polyorder)
        return smoothed, True
    except Exception as e:
        logger.warning(f"Smoothing failed: {e}")
        return reflectance, False


def remove_outliers(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    threshold: float = 5.0,  # Increased from 3.0 to be less aggressive
    library: str = 'default'
) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """
    Detect outliers but do not remove points. Apply gentle corrections to preserve spectral structure.
    
    Library-aware thresholds:
    - RELAB: minimal correction (high quality lab data)
    - RRUFF: moderate correction
    - USGS: robust correction but not destructive
    
    Returns modified reflectance (same length) and number of adjusted points.
    """
    # Library-specific thresholds
    library_thresholds = {
        'relab': 7.0,  # Very conservative for high-quality lab data
        'rruff': 5.0,  # Moderate
        'usgs': 5.0,   # Moderate
        'default': 5.0
    }
    
    threshold = library_thresholds.get(library.lower(), threshold)
    
    # Use Savitzky-Golay smoothing as the robust baseline
    try:
        from scipy.signal import savgol_filter
        window = 11
        if window % 2 == 0:
            window -= 1
        if len(reflectance) < window:
            # Not enough points to smooth; return original
            return wavelengths, reflectance, 0, {'method': 'savgol', 'adjusted': 0, 'library': library}
        smoothed = savgol_filter(reflectance, window, 3)
    except Exception:
        # fallback: use running median
        smoothed = pd.Series(reflectance).rolling(window=5, center=True, min_periods=1).median().to_numpy()

    # Compute robust deviation metric (MAD) and threshold
    resid = reflectance - smoothed
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad <= 0:
        mad = np.std(resid) if np.std(resid) > 0 else 1e-6
    limit = threshold * mad

    # Adjust values that deviate more than limit by pulling them toward the smoothed curve
    adjusted = 0
    ref_adj = reflectance.copy()
    large = np.abs(resid) > limit
    adjusted = int(np.sum(large))
    # Replace large deviations with smoothed +/- limit (gentler correction)
    ref_adj[large & (resid > 0)] = smoothed[large & (resid > 0)] + limit
    ref_adj[large & (resid < 0)] = smoothed[large & (resid < 0)] - limit

    stats = {'method': 'robust_smoothing', 'mad': float(mad), 'limit': float(limit), 'adjusted': adjusted, 'library': library, 'threshold_used': threshold}
    return wavelengths, ref_adj, adjusted, stats


def clean_spectrum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    remove_outliers_flag: bool = True,
    outlier_threshold: float = 5.0,
    min_points: int = 50,
    max_missing_fraction: float = 0.20,
    library: str = 'default',
    use_spectral_similarity: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Complete cleaning pipeline for a spectrum with library-aware parameters.
    
    Philosophy: Preserve spectral structure, correct artifacts, only remove clearly invalid data.
    
    Args:
        wavelengths: Wavelength array
        reflectance: Reflectance array
        window_length: Savitzky-Golay window length
        polyorder: Savitzky-Golay polynomial order
        remove_outliers_flag: Whether to apply outlier correction
        outlier_threshold: Outlier detection threshold (sigma)
        min_points: Minimum required points
        max_missing_fraction: Maximum allowed missing fraction
        library: Source library (relab, rruff, usgs) for library-aware cleaning
        use_spectral_similarity: Use spectral similarity for duplicate detection
        
    Returns:
        Tuple of (wavelengths_clean, reflectance_clean, report)
    """
    report = {
        'valid': False,
        'nan_removed': 0,
        'duplicates_removed': 0,
        'negatives_removed': 0,
        'outliers_removed': 0,
        'smoothing_successful': False,
        'validation_report': {},
        'noise_stats': {},
        'final_points': 0,
        'library': library,
        'distortion_metrics': {},
        'significantly_altered': False
    }

    # Store original for distortion monitoring
    ref_original = reflectance.copy() if len(reflectance) > 0 else reflectance

    # Initial validation
    is_valid, validation_report = validate_spectrum(
        wavelengths, reflectance,
        min_points=min_points,
        max_missing_fraction=max_missing_fraction
    )
    report['validation_report'] = validation_report

    if not is_valid:
        return wavelengths, reflectance, report

    wl, ref = wavelengths.copy(), reflectance.copy()

    wl, ref, nan_removed = remove_nans(wl, ref)
    report['nan_removed'] = nan_removed

    # Use spectral similarity-based duplicate detection
    wl, ref, dups_removed, dup_stats = remove_duplicates(wl, ref, use_spectral_similarity=use_spectral_similarity)
    report['duplicates_removed'] = dups_removed
    report['duplicate_stats'] = dup_stats

    wl, ref, negs_removed = remove_negative_values(wl, ref, clip_to_zero=True)
    report['negatives_clipped'] = negs_removed

    wl, ref = sort_by_wavelength(wl, ref)

    # Validate length consistency after cleaning steps
    if len(wl) != len(ref):
        logger.error(f"Length mismatch after cleaning: wavelengths={len(wl)}, reflectance={len(ref)}")
        report['validation_report']['issues'].append(f"Length mismatch after cleaning: {len(wl)} vs {len(ref)}")
        return wavelengths, reflectance, report

    if remove_outliers_flag:
        wl, ref_adj, outliers_adjusted, outlier_stats = remove_outliers(
            wl, ref, threshold=outlier_threshold, library=library
        )
        report['outliers_adjusted'] = outliers_adjusted
        report['outlier_stats'] = outlier_stats
        # use adjusted reflectance
        ref = ref_adj

    # After adjustments, do not remove points; validate for basic sanity
    is_valid_after, validation_after = validate_spectrum(
        wl, ref,
        min_points=min_points,
        max_missing_fraction=0.0
    )

    if not is_valid_after:
        # keep spectrum but mark invalid in report (allow higher-level to decide)
        report['validation_report_after'] = validation_after
        return wl, ref, report

    # Apply library-aware smoothing
    ref_smoothed, smoothing_success = apply_smoothing(ref, window_length, polyorder, library=library)
    report['smoothing_successful'] = smoothing_success

    # Compute spectral distortion metrics
    try:
        # Interpolate to same length for comparison if needed
        if len(ref_original) == len(ref_smoothed):
            distortion_metrics = compute_spectral_distortion_score(ref_original, ref_smoothed)
            variance_ratio = compute_variance_preservation_ratio(ref_original, ref_smoothed)
            distortion_metrics['variance_ratio'] = variance_ratio
            
            # Flag significantly altered spectra
            alteration_flags = flag_significantly_altered(distortion_metrics)
            report['distortion_metrics'] = distortion_metrics
            report['significantly_altered'] = alteration_flags['significantly_altered']
            report['alteration_reasons'] = alteration_flags['reasons']
    except Exception as e:
        logger.warning(f"Failed to compute distortion metrics: {e}")
        report['distortion_metrics'] = {}

    # Compute pre/post variance for diagnostics
    try:
        report['pre_variance'] = float(np.nanvar(ref))
        report['post_variance'] = float(np.nanvar(ref_smoothed))
    except Exception:
        report['pre_variance'] = None
        report['post_variance'] = None

    report['noise_stats'] = estimate_noise(ref_smoothed)

    report['valid'] = True
    report['final_points'] = len(wl)

    return wl, ref_smoothed, report
