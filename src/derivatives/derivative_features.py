"""
Derivative spectral features module.
"""
import numpy as np
import logging
from typing import Tuple, Dict, Optional


logger = logging.getLogger(__name__)



def compute_first_derivative(reflectance: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Compute first derivative of reflectance spectrum.
    
    Uses finite differences: dR/dλ ≈ (R[i+1] - R[i]) / Δλ
    
    Args:
        reflectance: Reflectance array
        wavelengths: Wavelength array (for accurate spacing), optional
        
    Returns:
        Tuple of (first_derivative, metadata)
    """
    try:
        valid_mask = ~np.isnan(reflectance)
        ref_valid = reflectance[valid_mask]
        
        if len(ref_valid) < 2:
            logger.warning("Insufficient points for derivative calculation")
            return np.full_like(reflectance, np.nan), {'method': 'none', 'valid': False}
        
        # Compute differences
        if wavelengths is not None and len(wavelengths) == len(reflectance):
            wl_valid = wavelengths[valid_mask]
            dlamb = np.diff(wl_valid)
            dref = np.diff(ref_valid)
            derivative_values = dref / dlamb
        else:
            derivative_values = np.diff(ref_valid)
        
        # Pad first derivative to match original length
        derivative_values = np.concatenate([[np.nan], derivative_values])
        
        # Reconstruct with NaNs
        derivative = np.full_like(reflectance, np.nan)
        derivative[valid_mask] = derivative_values[:len(ref_valid)]
        
        metadata = {
            'method': 'finite_difference',
            'valid': True,
            'order': 1,
            'min': float(np.nanmin(derivative_values)),
            'max': float(np.nanmax(derivative_values)),
            'mean': float(np.nanmean(derivative_values)),
            'std': float(np.nanstd(derivative_values)),
        }
        
        return derivative, metadata
    
    except Exception as e:
        logger.error(f"First derivative computation failed: {e}")
        return np.full_like(reflectance, np.nan), {'method': 'finite_difference', 'valid': False, 'error': str(e)}


def compute_second_derivative(reflectance: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Compute second derivative of reflectance spectrum.
    
    Uses finite differences: d²R/dλ² ≈ (R[i+1] - 2*R[i] + R[i-1]) / Δλ²
    
    Args:
        reflectance: Reflectance array
        wavelengths: Wavelength array (for accurate spacing), optional
        
    Returns:
        Tuple of (second_derivative, metadata)
    """
    try:
        valid_mask = ~np.isnan(reflectance)
        ref_valid = reflectance[valid_mask]
        
        if len(ref_valid) < 3:
            logger.warning("Insufficient points for second derivative calculation (< 3)")
            return np.full_like(reflectance, np.nan), {'method': 'none', 'valid': False}
        
        # Compute second differences
        if wavelengths is not None and len(wavelengths) == len(reflectance):
            wl_valid = wavelengths[valid_mask]
            
            # Use central differences where possible
            derivative_values = np.zeros(len(ref_valid))
            
            for i in range(1, len(ref_valid) - 1):
                dlamb_left = wl_valid[i] - wl_valid[i - 1]
                dlamb_right = wl_valid[i + 1] - wl_valid[i]
                dlamb_avg = (dlamb_left + dlamb_right) / 2
                
                ddr = ((ref_valid[i + 1] - ref_valid[i]) / dlamb_right - 
                       (ref_valid[i] - ref_valid[i - 1]) / dlamb_left)
                
                derivative_values[i] = ddr / dlamb_avg if dlamb_avg > 0 else 0
            
            # Forward/backward differences at edges
            derivative_values[0] = np.nan
            derivative_values[-1] = np.nan
        else:
            # Simple second differences
            second_diff = np.diff(ref_valid, n=2)
            derivative_values = np.concatenate([[np.nan], second_diff, [np.nan]])
        
        # Reconstruct with NaNs
        derivative = np.full_like(reflectance, np.nan)
        derivative[valid_mask] = derivative_values[:len(ref_valid)]
        
        valid_derivatives = derivative_values[~np.isnan(derivative_values)]
        
        metadata = {
            'method': 'finite_difference',
            'valid': True,
            'order': 2,
            'min': float(np.nanmin(valid_derivatives)) if len(valid_derivatives) > 0 else 0.0,
            'max': float(np.nanmax(valid_derivatives)) if len(valid_derivatives) > 0 else 0.0,
            'mean': float(np.nanmean(valid_derivatives)) if len(valid_derivatives) > 0 else 0.0,
            'std': float(np.nanstd(valid_derivatives)) if len(valid_derivatives) > 0 else 0.0,
        }
        
        return derivative, metadata
    
    except Exception as e:
        logger.error(f"Second derivative computation failed: {e}")
        return np.full_like(reflectance, np.nan), {'method': 'finite_difference', 'valid': False, 'error': str(e)}


def compute_derivatives(
    reflectance: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    orders: list = [1, 2]
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Compute multiple derivative orders.
    
    Args:
        reflectance: Reflectance array
        wavelengths: Wavelength array, optional
        orders: List of derivative orders to compute
        
    Returns:
        Dictionary mapping order -> (derivative_array, metadata)
    """
    results = {}
    
    if 1 in orders:
        first_deriv, first_meta = compute_first_derivative(reflectance, wavelengths)
        results[1] = (first_deriv, first_meta)
    
    if 2 in orders:
        second_deriv, second_meta = compute_second_derivative(reflectance, wavelengths)
        results[2] = (second_deriv, second_meta)
    
    return results
