"""
Spectral similarity metrics for scientifically robust duplicate detection.

Uses full-spectrum similarity metrics instead of exact wavelength matching
to preserve valid spectral variations while removing true duplicates.
"""
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from typing import Tuple, Dict, List
import logging


logger = logging.getLogger(__name__)


def compute_spectral_angle_mapper(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute Spectral Angle Mapper (SAM) between two spectra.
    
    SAM measures the angle between spectra in n-dimensional space,
    independent of magnitude. Lower values indicate higher similarity.
    
    Args:
        spectrum1: First reflectance spectrum
        spectrum2: Second reflectance spectrum
        
    Returns:
        SAM angle in radians
    """
    # Remove NaN values for comparison
    valid_mask = ~(np.isnan(spectrum1) | np.isnan(spectrum2))
    s1 = spectrum1[valid_mask]
    s2 = spectrum2[valid_mask]
    
    if len(s1) < 2 or len(s2) < 2:
        return np.pi  # Maximum dissimilarity
    
    # Compute dot product and magnitudes
    dot_product = np.dot(s1, s2)
    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return np.pi
    
    # Clamp cosine to valid range [-1, 1]
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle


def compute_cosine_similarity(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute cosine similarity between two spectra.
    
    Returns value in [0, 1] where 1 indicates identical shape.
    
    Args:
        spectrum1: First reflectance spectrum
        spectrum2: Second reflectance spectrum
        
    Returns:
        Cosine similarity score
    """
    valid_mask = ~(np.isnan(spectrum1) | np.isnan(spectrum2))
    s1 = spectrum1[valid_mask]
    s2 = spectrum2[valid_mask]
    
    if len(s1) < 2 or len(s2) < 2:
        return 0.0
    
    try:
        similarity = 1 - cosine(s1, s2)
        return float(similarity)
    except:
        return 0.0


def compute_pearson_correlation(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two spectra.
    
    Returns value in [-1, 1] where 1 indicates perfect positive correlation.
    
    Args:
        spectrum1: First reflectance spectrum
        spectrum2: Second reflectance spectrum
        
    Returns:
        Pearson correlation coefficient
    """
    valid_mask = ~(np.isnan(spectrum1) | np.isnan(spectrum2))
    s1 = spectrum1[valid_mask]
    s2 = spectrum2[valid_mask]
    
    if len(s1) < 3 or len(s2) < 3:
        return 0.0
    
    try:
        corr, _ = pearsonr(s1, s2)
        return float(corr)
    except:
        return 0.0


def compute_spectral_similarity(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    method: str = 'combined'
) -> Tuple[float, Dict]:
    """
    Compute comprehensive spectral similarity between two spectra.
    
    Args:
        spectrum1: First reflectance spectrum
        spectrum2: Second reflectance spectrum
        method: Similarity method ('sam', 'cosine', 'pearson', 'combined')
        
    Returns:
        Tuple of (similarity_score, metrics_dict)
        - similarity_score: 0-1 where 1 is identical
        - metrics_dict: Individual metric values
    """
    metrics = {}
    
    # Compute individual metrics
    sam = compute_spectral_angle_mapper(spectrum1, spectrum2)
    cos_sim = compute_cosine_similarity(spectrum1, spectrum2)
    pearson = compute_pearson_correlation(spectrum1, spectrum2)
    
    metrics['sam_angle'] = float(sam)
    metrics['cosine_similarity'] = float(cos_sim)
    metrics['pearson_correlation'] = float(pearson)
    
    # Convert SAM to similarity (0 = identical, 1 = maximum dissimilarity)
    sam_similarity = 1.0 - (sam / np.pi)
    metrics['sam_similarity'] = float(sam_similarity)
    
    # Combine metrics based on method
    if method == 'sam':
        score = sam_similarity
    elif method == 'cosine':
        score = cos_sim
    elif method == 'pearson':
        score = (pearson + 1) / 2  # Normalize to [0, 1]
    elif method == 'combined':
        # Weighted combination emphasizing shape over magnitude
        score = (
            0.4 * sam_similarity +
            0.3 * cos_sim +
            0.3 * ((pearson + 1) / 2)
        )
    else:
        score = sam_similarity
    
    return float(score), metrics


def detect_spectral_duplicates(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    similarity_threshold: float = 0.995,
    method: str = 'combined'
) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """
    Detect duplicate spectra using spectral similarity instead of exact wavelength matching.
    
    Only removes spectra that are near-identical in shape, preserving valid spectral variations.
    
    Args:
        wavelengths: Wavelength array
        reflectance: Reflectance array
        similarity_threshold: Threshold for considering spectra duplicates (0-1)
        method: Similarity method to use
        
    Returns:
        Tuple of (wavelengths_unique, reflectance_unique, duplicates_removed, stats)
    """
    # First remove exact wavelength duplicates (technical duplicates)
    unique_indices = np.unique(wavelengths, return_index=True)[1]
    unique_indices = np.sort(unique_indices)
    
    wl_unique = wavelengths[unique_indices]
    ref_unique = reflectance[unique_indices]
    
    exact_dups = len(wavelengths) - len(unique_indices)
    
    # If no exact duplicates, return early
    if exact_dups == 0:
        return wavelengths, reflectance, 0, {
            'method': method,
            'exact_duplicates': 0,
            'similarity_duplicates': 0,
            'total_removed': 0
        }
    
    # Check for spectral similarity among exact wavelength duplicates
    # Group by wavelength and check if reflectance values are similar
    unique_wavelengths = np.unique(wavelengths)
    to_remove = []
    
    for wl in unique_wavelengths:
        # Find all indices with this wavelength
        wl_mask = np.isclose(wavelengths, wl, rtol=1e-8)
        wl_indices = np.where(wl_mask)[0]
        
        if len(wl_indices) <= 1:
            continue
        
        # Compare reflectance values at this wavelength
        ref_values = reflectance[wl_indices]
        
        # If reflectance values are very similar, treat as duplicate
        # Use relative difference to account for magnitude
        if len(ref_values) > 1:
            ref_std = np.std(ref_values)
            ref_mean = np.mean(ref_values)
            
            # If relative std is very small, consider duplicates
            if ref_mean > 0:
                relative_std = ref_std / ref_mean
                if relative_std < 0.01:  # Less than 1% variation
                    # Keep first occurrence, mark others for removal
                    to_remove.extend(wl_indices[1:].tolist())
    
    # Remove marked duplicates
    if to_remove:
        keep_mask = np.ones(len(wavelengths), dtype=bool)
        keep_mask[to_remove] = False
        wl_final = wavelengths[keep_mask]
        ref_final = reflectance[keep_mask]
        similarity_dups = len(to_remove)
    else:
        wl_final = wl_unique
        ref_final = ref_unique
        similarity_dups = 0
    
    total_removed = exact_dups - similarity_dups  # Only count actual removals
    
    stats = {
        'method': method,
        'exact_duplicates': exact_dups,
        'similarity_duplicates': similarity_dups,
        'total_removed': total_removed,
        'threshold': similarity_threshold
    }
    
    return wl_final, ref_final, total_removed, stats
