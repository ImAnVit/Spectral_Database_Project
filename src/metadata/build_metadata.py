"""
Metadata database building module.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import uuid


logger = logging.getLogger(__name__)


def create_spectrum_metadata(
    spectrum_id: str,
    mineral_name: str,
    source_library: str,
    original_filename: str,
    raw_wavelengths: np.ndarray,
    cleaned_wavelengths: np.ndarray,
    processing_stages: Dict,
    quality_score: float = None,
    **kwargs
) -> Dict:
    """
    Create a metadata record for a processed spectrum.
    
    Args:
        spectrum_id: Unique spectrum identifier
        mineral_name: Name of mineral
        source_library: Source library (RELAB, RRUFF, USGS)
        original_filename: Original file name
        raw_wavelengths: Original wavelength array
        cleaned_wavelengths: Cleaned wavelength array
        processing_stages: Dictionary with stage-specific metadata
        **kwargs: Additional metadata fields
        
    Returns:
        Dictionary containing complete metadata
    """
    
    metadata = {
        'spectrum_id': spectrum_id,
        'mineral_name': mineral_name,
        'source_library': source_library,
        'original_filename': original_filename,
        'measurement_type': kwargs.get('measurement_type', 'reflectance'),
        'instrument': kwargs.get('instrument', 'unknown'),
        'collection_date': kwargs.get('collection_date', 'unknown'),
        'grain_size': kwargs.get('grain_size', 'unknown'),
        'sample_origin': kwargs.get('sample_origin', 'unknown'),
        'reference_source': kwargs.get('reference_source', ''),
        'crystal_system': kwargs.get('crystal_system', 'unknown'),
        'formula': kwargs.get('formula', ''),
        'space_group': kwargs.get('space_group', ''),
        'space_group_number': kwargs.get('space_group_number', ''),
    }
    
    # Wavelength statistics
    metadata.update({
        'wavelength_min': float(np.min(raw_wavelengths)) if len(raw_wavelengths) > 0 else np.nan,
        'wavelength_max': float(np.max(raw_wavelengths)) if len(raw_wavelengths) > 0 else np.nan,
        'num_points': len(cleaned_wavelengths),
    })
    # Optional diagnostics from processing_stages / kwargs
    # coverage from interpolation
    if 'interpolation' in processing_stages:
        interp = processing_stages['interpolation']
        metadata['coverage'] = float(interp.get('coverage', np.nan))
        metadata['interpolated_points'] = int(interp.get('output_points', 0))
    else:
        metadata['coverage'] = kwargs.get('coverage', np.nan)
        metadata['interpolated_points'] = kwargs.get('interpolated_points', None)

    # pre/post variance and adjustments
    metadata['pre_variance'] = kwargs.get('pre_variance', None)
    metadata['post_variance'] = kwargs.get('post_variance', None)
    metadata['negatives_clipped'] = kwargs.get('negatives_clipped', 0)
    metadata['outliers_adjusted'] = kwargs.get('outliers_adjusted', 0)
    
    # Processing steps
    if 'cleaning' in processing_stages:
        cleaning = processing_stages['cleaning']
        metadata.update({
            'smoothing_window': cleaning.get('window_length', 11),
            'smoothing_polynomial_order': cleaning.get('polyorder', 3),
        })
    
    if 'interpolation' in processing_stages:
        interp = processing_stages['interpolation']
        metadata['interpolation_method'] = interp.get('interpolation_method', 'linear')
    
    if 'normalization' in processing_stages:
        norm = processing_stages['normalization']
        metadata['normalization_method'] = norm.get('normalization_method', 'minmax')
    
    # Processing steps list
    processing_steps = []
    if 'cleaning' in processing_stages:
        processing_steps.append('cleaning')
    if 'interpolation' in processing_stages:
        processing_steps.append('interpolation')
    if 'normalization' in processing_stages:
        processing_steps.append('normalization')
    if 'continuum_removal' in processing_stages:
        processing_steps.append('continuum_removal')
    
    metadata['processing_steps'] = ','.join(processing_steps)
    
    # Quality flag
    quality_issues = []
    if 'cleaning' in processing_stages:
        if not processing_stages['cleaning'].get('valid', False):
            quality_issues.append('cleaning_failed')
    
    metadata['quality_flag'] = 'OK' if not quality_issues else ','.join(quality_issues)
    # Add quality_score if provided
    if quality_score is not None:
        metadata['quality_score'] = float(quality_score)
    else:
        metadata['quality_score'] = ''
    
    return metadata


def build_metadata_database(
    metadata_records: List[Dict],
    output_path: Path
) -> pd.DataFrame:
    """
    Build and save metadata database from spectrum records.
    
    Args:
        metadata_records: List of metadata dictionaries
        output_path: Path to save CSV file
        
    Returns:
        Pandas DataFrame with metadata
    """
    
    # Create DataFrame
    df = pd.DataFrame(metadata_records)
    
    # Define column order for consistent output
    column_order = [
        'spectrum_id', 'mineral_name', 'source_library', 'original_filename',
        'measurement_type', 'instrument', 'collection_date', 'grain_size',
        'sample_origin', 'reference_source', 'crystal_system', 'formula',
        'space_group', 'space_group_number', 'wavelength_min', 'wavelength_max',
        'num_points', 'processing_steps', 'smoothing_window',
        'smoothing_polynomial_order', 'interpolation_method', 'normalization_method',
        'quality_flag'
    ]
    
    # Reorder columns, adding any extras at the end
    existing_cols = [col for col in column_order if col in df.columns]
    extra_cols = [col for col in df.columns if col not in column_order]
    df = df[existing_cols + extra_cols]
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Metadata database saved: {output_path}")
    logger.info(f"Total records: {len(df)}")
    
    return df


def generate_summary_statistics(
    metadata_df: pd.DataFrame,
    output_path: Path
) -> pd.DataFrame:
    """
    Generate summary statistics from metadata database.
    
    Args:
        metadata_df: Metadata DataFrame
        output_path: Path to save CSV file
        
    Returns:
        DataFrame with summary statistics
    """
    
    summary = {
        'statistic': [],
        'value': []
    }
    
    # Total statistics
    summary['statistic'].append('total_spectra')
    summary['value'].append(len(metadata_df))
    
    summary['statistic'].append('spectra_accepted')
    summary['value'].append(len(metadata_df[metadata_df['quality_flag'] == 'OK']))
    
    summary['statistic'].append('spectra_rejected')
    summary['value'].append(len(metadata_df[metadata_df['quality_flag'] != 'OK']))
    
    # Per-mineral statistics
    mineral_counts = metadata_df['mineral_name'].value_counts()
    for mineral, count in mineral_counts.items():
        summary['statistic'].append(f'mineral_{mineral.lower().replace(" ", "_")}')
        summary['value'].append(int(count))
    
    # Per-library statistics
    library_counts = metadata_df['source_library'].value_counts()
    for library, count in library_counts.items():
        summary['statistic'].append(f'library_{library.lower()}')
        summary['value'].append(int(count))
    
    # Wavelength statistics
    summary['statistic'].append('wavelength_min_overall')
    summary['value'].append(float(metadata_df['wavelength_min'].min()))
    
    summary['statistic'].append('wavelength_max_overall')
    summary['value'].append(float(metadata_df['wavelength_max'].max()))
    
    summary['statistic'].append('avg_spectrum_points')
    summary['value'].append(float(metadata_df['num_points'].mean()))
    
    # Create DataFrame
    df_summary = pd.DataFrame(summary)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Summary statistics saved: {output_path}")
    
    return df_summary
