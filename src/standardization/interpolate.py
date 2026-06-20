"""
Spectrum interpolation module.
"""
import numpy as np
from scipy.interpolate import interp1d
import logging
from typing import Tuple, Dict


logger = logging.getLogger(__name__)


def interpolate_spectrum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    wavelength_min: float = 400.0,
    wavelength_max: float = 2500.0,
    wavelength_step: float = 1.0,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Interpolate spectrum onto a common wavelength grid.
    
    Args:
        wavelengths: Original wavelength array
        reflectance: Original reflectance array
        wavelength_min: Minimum wavelength of output grid
        wavelength_max: Maximum wavelength of output grid
        wavelength_step: Step size of output grid
        method: Interpolation method ('linear', 'cubic', 'quadratic')
        
    Returns:
        Tuple of (interpolated_wavelengths, interpolated_reflectance, metadata)
    """
    try:
        # Create target wavelength grid
        target_wavelengths = np.arange(
            wavelength_min,
            wavelength_max + wavelength_step,
            wavelength_step
        )
        
        # Check for sufficient data points
        if len(wavelengths) < 2:
            raise ValueError(f"Insufficient data points for interpolation: {len(wavelengths)}")
        
        # Ensure input wavelengths are sorted and unique for interpolation
        sort_idx = np.argsort(wavelengths)
        x = np.asarray(wavelengths)[sort_idx]
        y = np.asarray(reflectance)[sort_idx]
        # remove duplicates by averaging
        if len(x) > 1:
            uniq_x, uniq_inds = np.unique(np.round(x,8), return_inverse=True)
            y_agg = np.zeros_like(uniq_x, dtype=float)
            counts = np.zeros_like(uniq_x, dtype=int)
            for i, u in enumerate(uniq_inds):
                y_agg[u] += y[i]
                counts[u] += 1
            y_agg = y_agg / counts
            x = uniq_x.astype(float)
            y = y_agg.astype(float)

        # Create interpolation function
        kind = method if method in ['linear', 'cubic', 'quadratic'] else 'linear'
        try:
            f = interp1d(
                x,
                y,
                kind=kind,
                bounds_error=False,
                # Fill outside measured range with edge values (preserve shape without NaNs)
                fill_value=(float(y[0]), float(y[-1])) if len(y) > 0 else np.nan
            )
        except Exception as e:
            logger.error(f"Failed to construct interpolator: {e}")
            raise

        # Interpolate
        interpolated_reflectance = f(target_wavelengths)

        # Calculate coverage statistics: now should be full grid (no NaNs)
        valid_mask = np.isfinite(interpolated_reflectance)
        coverage = np.sum(valid_mask) / len(target_wavelengths)

        metadata = {
            'interpolation_method': method,
            'wavelength_min': float(wavelength_min),
            'wavelength_max': float(wavelength_max),
            'wavelength_step': float(wavelength_step),
            'output_points': len(target_wavelengths),
            'coverage': float(coverage),
            'nan_count': int(np.sum(~valid_mask)),
            'extrapolated': int(np.any((target_wavelengths < x.min()) | (target_wavelengths > x.max()))) if len(x)>0 else 1
        }
        
        logger.info(f"Interpolation successful: {len(wavelengths)} -> {len(target_wavelengths)} points, "
                   f"coverage: {coverage:.1%}")
        
        return target_wavelengths, interpolated_reflectance, metadata
    
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise
