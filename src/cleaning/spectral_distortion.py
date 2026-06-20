"""
Spectral distortion monitoring metrics to track preprocessing impact on physical fidelity.

Measures how much preprocessing alters spectral shapes and preserves physical meaning.
"""
import numpy as np
from typing import Tuple, Dict, Optional
import logging


logger = logging.getLogger(__name__)


def compute_sam_before_after(
    reflectance_before: np.ndarray,
    reflectance_after: np.ndarray
) -> float:
    """
    Compute Spectral Angle Mapper between before and after preprocessing.
    
    Lower values indicate less distortion (better preservation).
    
    Args:
        reflectance_before: Original reflectance spectrum
        reflectance_after: Processed reflectance spectrum
        
    Returns:
        SAM angle in radians
    """
    from .spectral_similarity import compute_spectral_angle_mapper
    return compute_spectral_angle_mapper(reflectance_before, reflectance_after)


def compute_spectral_distortion_score(
    reflectance_before: np.ndarray,
    reflectance_after: np.ndarray
) -> Dict:
    """
    Compute comprehensive distortion score between original and processed spectrum.
    
    Returns multiple metrics to quantify preprocessing impact.
    
    Args:
        reflectance_before: Original reflectance spectrum
        reflectance_after: Processed reflectance spectrum
        
    Returns:
        Dictionary with distortion metrics
    """
    valid_mask = ~(np.isnan(reflectance_before) | np.isnan(reflectance_after))
    ref_before = reflectance_before[valid_mask]
    ref_after = reflectance_after[valid_mask]
    
    if len(ref_before) < 2:
        return {
            'sam_angle': np.pi,
            'cosine_similarity': 0.0,
            'pearson_correlation': 0.0,
            'rmse': float('inf'),
            'mae': float('inf'),
            'relative_change': 1.0,
            'distortion_score': 1.0
        }
    
    # SAM
    sam = compute_sam_before_after(ref_before, ref_after)
    
    # Cosine similarity
    from .spectral_similarity import compute_cosine_similarity
    cos_sim = compute_cosine_similarity(ref_before, ref_after)
    
    # Pearson correlation
    from .spectral_similarity import compute_pearson_correlation
    pearson = compute_pearson_correlation(ref_before, ref_after)
    
    # RMSE
    rmse = np.sqrt(np.mean((ref_before - ref_after) ** 2))
    
    # MAE
    mae = np.mean(np.abs(ref_before - ref_after))
    
    # Relative change (normalized by original magnitude)
    ref_before_norm = ref_before / (np.mean(np.abs(ref_before)) + 1e-10)
    ref_after_norm = ref_after / (np.mean(np.abs(ref_before)) + 1e-10)
    relative_change = np.mean(np.abs(ref_before_norm - ref_after_norm))
    
    # Combined distortion score (0 = no distortion, 1 = maximum distortion)
    sam_normalized = sam / np.pi
    distortion_score = (
        0.3 * sam_normalized +
        0.3 * (1 - cos_sim) +
        0.2 * (1 - (pearson + 1) / 2) +
        0.2 * relative_change
    )
    
    return {
        'sam_angle': float(sam),
        'cosine_similarity': float(cos_sim),
        'pearson_correlation': float(pearson),
        'rmse': float(rmse),
        'mae': float(mae),
        'relative_change': float(relative_change),
        'distortion_score': float(np.clip(distortion_score, 0.0, 1.0))
    }


def compute_variance_preservation_ratio(
    reflectance_before: np.ndarray,
    reflectance_after: np.ndarray
) -> float:
    """
    Compute ratio of variance after vs before preprocessing.
    
    Values close to 1.0 indicate variance is preserved.
    Values << 1.0 indicate over-smoothing (variance loss).
    Values >> 1.0 indicate amplification.
    
    Args:
        reflectance_before: Original reflectance spectrum
        reflectance_after: Processed reflectance spectrum
        
    Returns:
        Variance ratio (post / pre)
    """
    valid_mask = ~(np.isnan(reflectance_before) | np.isnan(reflectance_after))
    ref_before = reflectance_before[valid_mask]
    ref_after = reflectance_after[valid_mask]
    
    if len(ref_before) < 2:
        return 0.0
    
    var_before = np.var(ref_before)
    var_after = np.var(ref_after)
    
    if var_before == 0:
        return 1.0 if var_after == 0 else 0.0
    
    return var_after / var_before


def flag_significantly_altered(
    distortion_metrics: Dict,
    sam_threshold: float = 0.1,  # radians
    distortion_threshold: float = 0.15,
    variance_ratio_range: Tuple[float, float] = (0.5, 2.0)
) -> Dict:
    """
    Flag spectra that were significantly altered by preprocessing.
    
    Args:
        distortion_metrics: Distortion metrics from compute_spectral_distortion_score
        sam_threshold: Maximum acceptable SAM angle (radians)
        distortion_threshold: Maximum acceptable combined distortion score
        variance_ratio_range: Acceptable range for variance ratio (min, max)
        
    Returns:
        Dictionary with flags and reasons
    """
    flags = {
        'significantly_altered': False,
        'reasons': []
    }
    
    # Check SAM
    if distortion_metrics['sam_angle'] > sam_threshold:
        flags['significantly_altered'] = True
        flags['reasons'].append(f"High SAM angle: {distortion_metrics['sam_angle']:.4f} > {sam_threshold}")
    
    # Check combined distortion score
    if distortion_metrics['distortion_score'] > distortion_threshold:
        flags['significantly_altered'] = True
        flags['reasons'].append(f"High distortion score: {distortion_metrics['distortion_score']:.4f} > {distortion_threshold}")
    
    # Check variance preservation (if available)
    if 'variance_ratio' in distortion_metrics:
        var_ratio = distortion_metrics['variance_ratio']
        if var_ratio < variance_ratio_range[0]:
            flags['significantly_altered'] = True
            flags['reasons'].append(f"Over-smoothed: variance ratio {var_ratio:.4f} < {variance_ratio_range[0]}")
        elif var_ratio > variance_ratio_range[1]:
            flags['significantly_altered'] = True
            flags['reasons'].append(f"Over-amplified: variance ratio {var_ratio:.4f} > {variance_ratio_range[1]}")
    
    return flags
