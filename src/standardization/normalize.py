"""
Spectrum normalization module.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging
from typing import Tuple, Dict


logger = logging.getLogger(__name__)


def normalize_minmax(reflectance: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict]:
    """
    Apply min-max normalization (scales to [0, 1] by default).
    
    Args:
        reflectance: Reflectance array
        feature_range: Target range (min, max)
        
    Returns:
        Tuple of (normalized_reflectance, metadata)
    """
    valid_mask = ~np.isnan(reflectance)
    reflectance_valid = reflectance[valid_mask].reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=feature_range)
    reflectance_valid_norm = scaler.fit_transform(reflectance_valid).flatten()
    
    # Reconstruct array with NaNs in original positions
    normalized = np.full_like(reflectance, np.nan)
    normalized[valid_mask] = reflectance_valid_norm
    
    metadata = {
        'normalization_method': 'minmax',
        'feature_range': feature_range,
        'original_min': float(np.nanmin(reflectance)),
        'original_max': float(np.nanmax(reflectance)),
        'normalized_min': float(np.nanmin(normalized)),
        'normalized_max': float(np.nanmax(normalized)),
    }
    
    return normalized, metadata


def normalize_stddev(reflectance: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Apply z-score normalization (standardization).
    
    Args:
        reflectance: Reflectance array
        
    Returns:
        Tuple of (normalized_reflectance, metadata)
    """
    valid_mask = ~np.isnan(reflectance)
    reflectance_valid = reflectance[valid_mask].reshape(-1, 1)
    
    scaler = StandardScaler()
    reflectance_valid_norm = scaler.fit_transform(reflectance_valid).flatten()
    
    # Reconstruct array with NaNs in original positions
    normalized = np.full_like(reflectance, np.nan)
    normalized[valid_mask] = reflectance_valid_norm
    
    metadata = {
        'normalization_method': 'stddev',
        'original_mean': float(np.nanmean(reflectance)),
        'original_std': float(np.nanstd(reflectance)),
        'normalized_mean': float(np.nanmean(normalized)),
        'normalized_std': float(np.nanstd(normalized)),
    }
    
    return normalized, metadata


def normalize_max(reflectance: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize by maximum value (reflectance / max).
    
    Args:
        reflectance: Reflectance array
        
    Returns:
        Tuple of (normalized_reflectance, metadata)
    """
    valid_mask = ~np.isnan(reflectance)
    reflectance_valid = reflectance[valid_mask]
    
    max_val = np.nanmax(reflectance)
    if max_val > 0:
        normalized = reflectance / max_val
    else:
        normalized = reflectance.copy()
        logger.warning("Maximum reflectance is zero or negative, normalization skipped")
    
    metadata = {
        'normalization_method': 'max',
        'maximum_value': float(max_val),
        'normalized_max': float(np.nanmax(normalized)),
    }
    
    return normalized, metadata


def normalize_robust(reflectance: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Apply robust scaling (less affected by outliers).
    
    Uses median and interquartile range.
    
    Args:
        reflectance: Reflectance array
        
    Returns:
        Tuple of (normalized_reflectance, metadata)
    """
    valid_mask = ~np.isnan(reflectance)
    reflectance_valid = reflectance[valid_mask].reshape(-1, 1)
    
    scaler = RobustScaler()
    reflectance_valid_norm = scaler.fit_transform(reflectance_valid).flatten()
    
    # Reconstruct array with NaNs in original positions
    normalized = np.full_like(reflectance, np.nan)
    normalized[valid_mask] = reflectance_valid_norm
    
    metadata = {
        'normalization_method': 'robust_scale',
        'median': float(np.nanmedian(reflectance)),
        'q1': float(np.nanpercentile(reflectance, 25)),
        'q3': float(np.nanpercentile(reflectance, 75)),
    }
    
    return normalized, metadata


def normalize_spectrum(
    reflectance: np.ndarray,
    method: str = 'minmax'
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize spectrum using specified method.
    
    Args:
        reflectance: Reflectance array
        method: Normalization method ('minmax', 'stddev', 'max', 'robust_scale')
        
    Returns:
        Tuple of (normalized_reflectance, metadata)
    """
    try:
        if method == 'minmax':
            return normalize_minmax(reflectance)
        elif method == 'stddev':
            return normalize_stddev(reflectance)
        elif method == 'max':
            return normalize_max(reflectance)
        elif method == 'robust_scale':
            return normalize_robust(reflectance)
        else:
            logger.warning(f"Unknown normalization method: {method}, using minmax")
            return normalize_minmax(reflectance)
    
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise
