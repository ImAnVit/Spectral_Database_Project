"""
Continuum removal module for spectral analysis.
"""
import numpy as np
from scipy.spatial import ConvexHull
import logging
from typing import Tuple, Dict


logger = logging.getLogger(__name__)


def compute_convex_hull_continuum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    """
    Compute spectral continuum using convex hull method.
    
    The continuum is the upper envelope of the spectrum, calculated
    using the convex hull of wavelength-reflectance points.
    
    Args:
        wavelengths: Array of wavelengths
        reflectance: Array of reflectance values
        
    Returns:
        Tuple of (continuum_reflectance, metadata)
    """
    try:
        # Remove NaN values
        valid_mask = ~np.isnan(reflectance)
        wl_valid = wavelengths[valid_mask]
        ref_valid = reflectance[valid_mask]
        
        if len(wl_valid) < 3:
            logger.warning("Insufficient points for convex hull (< 3 points)")
            return reflectance, {'method': 'none', 'valid': False}
        
        # Stack wavelengths and reflectance for convex hull
        points = np.column_stack([wl_valid, ref_valid])
        
        # Compute convex hull
        hull = ConvexHull(points)
        
        # Get vertices (indices of hull points)
        hull_indices = hull.vertices
        hull_points = points[hull_indices]
        
        # Sort by wavelength
        sorted_indices = np.argsort(hull_points[:, 0])
        hull_points_sorted = hull_points[sorted_indices]
        
        # Interpolate continuum to all wavelengths
        continuum_valid = np.interp(
            wl_valid,
            hull_points_sorted[:, 0],
            hull_points_sorted[:, 1],
            left=hull_points_sorted[0, 1],
            right=hull_points_sorted[-1, 1]
        )
        
        # Reconstruct continuum with NaNs in original positions
        continuum = np.full_like(reflectance, np.nan)
        continuum[valid_mask] = continuum_valid
        
        # Validate continuum (should be >= reflectance)
        invalid_continuum = np.sum(continuum_valid < ref_valid)
        
        metadata = {
            'method': 'convex_hull',
            'valid': True,
            'hull_points': len(hull_points),
            'invalid_points': int(invalid_continuum),
            'continuum_min': float(np.min(continuum_valid)),
            'continuum_max': float(np.max(continuum_valid)),
        }
        
        return continuum, metadata
    
    except Exception as e:
        logger.error(f"Convex hull continuum computation failed: {e}")
        return reflectance, {'method': 'convex_hull', 'valid': False, 'error': str(e)}


def remove_continuum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    method: str = 'convex_hull'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform continuum removal on spectrum.
    
    Continuum removal divides the reflectance by the continuum to enhance
    absorption features and normalize the spectrum.
    
    Args:
        wavelengths: Array of wavelengths
        reflectance: Array of reflectance values
        method: Continuum computation method ('convex_hull')
        
    Returns:
        Tuple of (continuum_removed_reflectance, continuum, metadata)
    """
    try:
        if method == 'convex_hull':
            continuum, continuum_meta = compute_convex_hull_continuum(wavelengths, reflectance)
        else:
            logger.warning(f"Unknown continuum method: {method}, using convex_hull")
            continuum, continuum_meta = compute_convex_hull_continuum(wavelengths, reflectance)
        
        if not continuum_meta.get('valid', False):
            # Return original spectrum if continuum computation failed
            return reflectance, reflectance, continuum_meta
        
        # Divide reflectance by continuum
        valid_mask = ~np.isnan(reflectance)
        continuum_removed = np.full_like(reflectance, np.nan)
        
        continuum_valid = continuum[valid_mask]
        ref_valid = reflectance[valid_mask]
        
        # Avoid division by zero
        nonzero_mask = continuum_valid > 0
        continuum_removed_valid = np.full_like(ref_valid, np.nan)
        continuum_removed_valid[nonzero_mask] = ref_valid[nonzero_mask] / continuum_valid[nonzero_mask]
        
        continuum_removed[valid_mask] = continuum_removed_valid
        
        # Statistics
        valid_cr_mask = ~np.isnan(continuum_removed_valid)
        cr_stats = {
            'method': method,
            'valid': True,
            'continuum_points': int(np.sum(valid_mask)),
            'cr_min': float(np.nanmin(continuum_removed_valid[valid_cr_mask])) if np.any(valid_cr_mask) else 0.0,
            'cr_max': float(np.nanmax(continuum_removed_valid[valid_cr_mask])) if np.any(valid_cr_mask) else 0.0,
            'cr_mean': float(np.nanmean(continuum_removed_valid[valid_cr_mask])) if np.any(valid_cr_mask) else 0.0,
            **continuum_meta,
        }
        
        return continuum_removed, continuum, cr_stats
    
    except Exception as e:
        logger.error(f"Continuum removal failed: {e}")
        return reflectance, reflectance, {'method': method, 'valid': False, 'error': str(e)}
