"""
Master preprocessing pipeline.

Orchestrates the complete spectral database preprocessing workflow:
1. Discover raw spectra
2. Cleaning and quality control
3. Standardization (interpolation + normalization)
4. Continuum removal
5. Derivative generation
6. Metadata database building
7. Report generation
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import json
import uuid

from src.config import ConfigManager
from src.utils import setup_logger
from src.parsers.spectrum_parser import parse_spectrum
from src.cleaning.clean_spectra import clean_spectrum
from src.standardization.interpolate import interpolate_spectrum
from src.standardization.normalize import normalize_spectrum
from src.continuum.continuum_removal import remove_continuum
from src.derivatives.derivative_features import compute_derivatives
from src.metadata.build_metadata import create_spectrum_metadata, build_metadata_database, generate_summary_statistics
from src.reporting.generate_report import generate_markdown_report


logger = logging.getLogger(__name__)


class SpectralPreprocessor:
    """Main preprocessing pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            __name__,
            log_file=self.config.log_file,
            level=self.config.log_level,
            log_format=self.config.log_format
        )
        
        # Statistics tracking
        self.processing_stats = {
            'total_processed': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'mineral_counts': {},
            'library_counts': {},
            'library_processed': {'relab': 0, 'rruff': 0, 'usgs': 0},
            'library_accepted': {'relab': 0, 'rruff': 0, 'usgs': 0},
            'library_rejected': {'relab': 0, 'rruff': 0, 'usgs': 0},
            'library_discovered': {'relab': 0, 'rruff': 0, 'usgs': 0},
        }
        
        self.cleaning_stats = {
            'total_nans_removed': 0,
            'total_duplicates_removed': 0,
            'total_outliers_removed': 0,
            'total_outliers_adjusted': 0,
            'total_negatives_clipped': 0,
            'total_smoothing_applied': 0,
            'cleaning_failures': 0,
            'spectral_distortion_stats': [],
            'significantly_altered_count': 0,
            'library_cleaning_stats': {'relab': {}, 'rruff': {}, 'usgs': {}}
        }
        
        self.wavelength_stats = {
            'min_wavelength': float('inf'),
            'max_wavelength': float('-inf'),
            'coverage_values': [],
        }
        
        self.metadata_records = []
        self.error_log = []
        # Rejection records for diagnostics
        self.rejection_records = []
    
    def _record_rejection(self, *, file_path: str, library: str, stage: str, reason: str, n_points: int = None, wavelength_min: float = None, wavelength_max: float = None, missing_count: int = None, quality_score: float = None, mineral: str = None) -> None:
        """Record a structured rejection entry for diagnostics.

        Fields recorded:
        - file_path, library, stage, reason
        - n_points, wavelength_min, wavelength_max, missing_count, quality_score, mineral
        """
        entry = {
            'file_path': file_path,
            'library': library,
            'stage': stage,
            'reason': reason,
            'n_points': int(n_points) if n_points is not None else None,
            'wavelength_min': float(wavelength_min) if wavelength_min is not None else None,
            'wavelength_max': float(wavelength_max) if wavelength_max is not None else None,
            'missing_count': int(missing_count) if missing_count is not None else None,
            'quality_score': float(quality_score) if quality_score is not None else None,
            'mineral': mineral
        }
        self.rejection_records.append(entry)

    def discover_spectra(self) -> dict:
        """Discover all raw spectrum files organized by library and mineral."""
        spectra_by_library = {'relab': {}, 'rruff': {}, 'usgs': {}}
        spectrum_extensions = ('*.csv', '*.txt', '*.asc', '*.tab')

        for library, path_str in self.config.raw_data_paths.items():
            base_path = Path(path_str)
            if not base_path.exists():
                self.logger.warning(f"Raw data path does not exist: {path_str}")
                continue

            for mineral_dir in base_path.iterdir():
                if mineral_dir.is_dir():
                    mineral = mineral_dir.name
                    spectra = []
                    for pattern in spectrum_extensions:
                        spectra.extend(mineral_dir.glob(f'**/{pattern}'))

                    if spectra:
                        if mineral not in spectra_by_library[library]:
                            spectra_by_library[library][mineral] = []
                        spectra_by_library[library][mineral].extend(spectra)

            library_count = sum(len(specs) for specs in spectra_by_library[library].values())
            self.processing_stats['library_discovered'][library] = library_count
            self.logger.info(
                f"Discovered {library_count} {library.upper()} spectra in {path_str}"
            )
            print(f"RELAB files found: {library_count}" if library == 'relab'
                  else f"{library.upper()} files found: {library_count}")

        total_spectra = sum(len(specs) for lib in spectra_by_library.values()
                           for specs in lib.values())
        self.logger.info(f"Discovered {total_spectra} spectra total")

        return spectra_by_library
    
    def process_spectrum(
        self,
        filepath: Path,
        mineral_name: str,
        source_library: str,
        spectrum_id: str = None
    ) -> tuple:
        """
        Process a single spectrum through the complete pipeline.
        
        Returns:
            (success, metadata_record, processed_data)
        """
        if spectrum_id is None:
            spectrum_id = f"{mineral_name}_{filepath.stem}_{uuid.uuid4().hex[:8]}"
        
        processed_data = {}
        processing_stages = {}
        
        try:
            # Parse raw spectrum using unified parser
            wl_raw, ref_raw, fmt, parsed_valid, parse_error = parse_spectrum(
                filepath,
                wl_min=self.config.wavelength_min,
                wl_max=self.config.wavelength_max,
                wl_step=self.config.wavelength_step,
                min_points=10
            )
            processed_data['wavelengths_raw'] = wl_raw
            processed_data['reflectance_raw'] = ref_raw
            processed_data['parser_format'] = fmt

            # Check parse validity / errors
            if not parsed_valid:
                reason = f"parse_error: {parse_error or 'parsing_failed'}"
                self.logger.warning(f"Parsing failed for {filepath}: {parse_error}")
                self.error_log.append(f"{filepath}: {reason}")
                self.cleaning_stats['cleaning_failures'] += 1
                self._record_rejection(
                    file_path=str(filepath),
                    library=source_library,
                    stage='parsing',
                    reason=reason,
                    n_points=int(wl_raw.size) if hasattr(wl_raw, 'size') else 0,
                    wavelength_min=float(wl_raw.min()) if hasattr(wl_raw, 'size') and wl_raw.size>0 else None,
                    wavelength_max=float(wl_raw.max()) if hasattr(wl_raw, 'size') and wl_raw.size>0 else None,
                    missing_count=int(np.sum(np.isnan(ref_raw))) if hasattr(ref_raw, 'size') else None,
                    mineral=mineral_name
                )
                return False, None, None

            # GLOBAL VALIDATION after parsing
            try:
                from src.cleaning.validate_spectra import validate_parsed_spectrum
                valid_after_parse, parse_report = validate_parsed_spectrum(wl_raw, ref_raw, min_points=10)
                if not valid_after_parse:
                    reason = f"validation_failed: {parse_report.get('issues', [])}"
                    self.logger.warning(f"Validation failed after parsing for {filepath}: {parse_report.get('issues', [])}")
                    self.error_log.append(f"{filepath}: {reason}")
                    self.cleaning_stats['cleaning_failures'] += 1
                    self._record_rejection(
                        file_path=str(filepath),
                        library=source_library,
                        stage='validation',
                        reason=reason,
                        n_points=int(np.sum(~np.isnan(wl_raw))) if wl_raw is not None else 0,
                        wavelength_min=float(np.nanmin(wl_raw)) if wl_raw is not None and wl_raw.size>0 else None,
                        wavelength_max=float(np.nanmax(wl_raw)) if wl_raw is not None and wl_raw.size>0 else None,
                        missing_count=int(np.sum(np.isnan(ref_raw))) if ref_raw is not None else None,
                        mineral=mineral_name
                    )
                    return False, None, None
            except Exception as e:
                reason = f"validation_exception: {e}"
                self.logger.error(f"Validation exception for {filepath}: {e}")
                self.error_log.append(f"{filepath}: {reason}")
                self.cleaning_stats['cleaning_failures'] += 1
                self._record_rejection(
                    file_path=str(filepath),
                    library=source_library,
                    stage='validation',
                    reason=reason,
                    n_points=0,
                    wavelength_min=None,
                    wavelength_max=None,
                    missing_count=None,
                    mineral=mineral_name
                )
                return False, None, None

            # STAGE 1: Cleaning (with library-aware parameters)
            wl_clean, ref_clean, cleaning_report = clean_spectrum(
                wl_raw, ref_raw,
                window_length=self.config.savgol_window,
                polyorder=self.config.savgol_polyorder,
                remove_outliers_flag=self.config.detect_outliers,
                outlier_threshold=self.config.outlier_threshold,
                min_points=self.config.min_spectrum_points,
                max_missing_fraction=self.config.max_missing_fraction,
                library=source_library,
                use_spectral_similarity=True
            )

            # Update cleaning diagnostics
            self.cleaning_stats['total_nans_removed'] += cleaning_report.get('nan_removed', 0)
            self.cleaning_stats['total_duplicates_removed'] += cleaning_report.get('duplicates_removed', 0)
            # count adjusted outliers and clipped negatives
            self.cleaning_stats['total_outliers_adjusted'] += cleaning_report.get('outliers_adjusted', 0)
            self.cleaning_stats['total_negatives_clipped'] += cleaning_report.get('negatives_clipped', 0)
            if cleaning_report.get('smoothing_successful'):
                self.cleaning_stats['total_smoothing_applied'] += 1
            
            # Track spectral distortion metrics
            if cleaning_report.get('distortion_metrics'):
                self.cleaning_stats['spectral_distortion_stats'].append({
                    'library': source_library,
                    'mineral': mineral_name,
                    'spectrum_id': spectrum_id,
                    **cleaning_report['distortion_metrics']
                })
            
            # Track significantly altered spectra
            if cleaning_report.get('significantly_altered'):
                self.cleaning_stats['significantly_altered_count'] += 1
            
            # Track per-library cleaning statistics
            lib_stats = self.cleaning_stats['library_cleaning_stats'].setdefault(source_library, {})
            lib_stats['nans_removed'] = lib_stats.get('nans_removed', 0) + cleaning_report.get('nan_removed', 0)
            lib_stats['duplicates_removed'] = lib_stats.get('duplicates_removed', 0) + cleaning_report.get('duplicates_removed', 0)
            lib_stats['outliers_adjusted'] = lib_stats.get('outliers_adjusted', 0) + cleaning_report.get('outliers_adjusted', 0)
            lib_stats['negatives_clipped'] = lib_stats.get('negatives_clipped', 0) + cleaning_report.get('negatives_clipped', 0)
            lib_stats['processed_count'] = lib_stats.get('processed_count', 0) + 1

            if not cleaning_report['valid']:
                issues = cleaning_report.get('validation_report', {}).get('issues', [])
                reason = f"cleaning_failed: {issues}"
                self.logger.warning(f"Cleaning failed for {filepath}: {issues}")
                self.cleaning_stats['cleaning_failures'] += 1
                self._record_rejection(
                    file_path=str(filepath),
                    library=source_library,
                    stage='cleaning',
                    reason=reason,
                    n_points=int(cleaning_report.get('final_points', 0)),
                    wavelength_min=float(np.nanmin(wl_raw)) if wl_raw is not None and wl_raw.size>0 else None,
                    wavelength_max=float(np.nanmax(wl_raw)) if wl_raw is not None and wl_raw.size>0 else None,
                    missing_count=int(cleaning_report.get('nan_removed', 0)),
                    mineral=mineral_name
                )
                return False, None, None
            
            processed_data['wavelengths_clean'] = wl_clean
            processed_data['reflectance_clean'] = ref_clean
            processing_stages['cleaning'] = {
                'window_length': self.config.savgol_window,
                'polyorder': self.config.savgol_polyorder,
                'valid': cleaning_report['valid'],
                'negatives_clipped': cleaning_report.get('negatives_clipped', 0),
                'outliers_adjusted': cleaning_report.get('outliers_adjusted', 0),
                'pre_variance': cleaning_report.get('pre_variance', None),
                'post_variance': cleaning_report.get('post_variance', None),
                'library': source_library,
                'distortion_metrics': cleaning_report.get('distortion_metrics', {}),
                'significantly_altered': cleaning_report.get('significantly_altered', False)
            }
            
            # Validate length consistency after cleaning
            if len(wl_clean) != len(ref_clean):
                self.logger.error(f"Length mismatch after cleaning: wavelengths={len(wl_clean)}, reflectance={len(ref_clean)}")
                self._record_rejection(
                    file_path=str(filepath),
                    library=source_library,
                    stage='cleaning_length_validation',
                    reason=f'Length mismatch after cleaning: {len(wl_clean)} vs {len(ref_clean)}',
                    n_points=len(wl_clean),
                    mineral=mineral_name
                )
                return False, None, None
            
            # Save cleaned spectrum
            cleaned_dir = Path(self.config.output_paths['cleaned']) / mineral_name
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            self._save_spectrum(wl_clean, ref_clean, cleaned_dir / f"{spectrum_id}_cleaned.parquet")
            
            # STAGE 2: Standardization - Interpolation
            wl_interp, ref_interp, interp_meta = interpolate_spectrum(
                wl_clean, ref_clean,
                wavelength_min=self.config.wavelength_min,
                wavelength_max=self.config.wavelength_max,
                wavelength_step=self.config.wavelength_step,
                method=self.config.interpolation_method
            )
            
            # Validate length consistency after interpolation
            if wl_interp is not None and ref_interp is not None:
                if len(wl_interp) != len(ref_interp):
                    self.logger.error(f"Length mismatch after interpolation: wavelengths={len(wl_interp)}, reflectance={len(ref_interp)}")
                    self._record_rejection(
                        file_path=str(filepath),
                        library=source_library,
                        stage='interpolation_length_validation',
                        reason=f'Length mismatch after interpolation: {len(wl_interp)} vs {len(ref_interp)}',
                        n_points=len(wl_interp),
                        mineral=mineral_name
                    )
                    return False, None, None
            
            processed_data['wavelengths_interp'] = wl_interp
            processed_data['reflectance_interp'] = ref_interp
            processing_stages['interpolation'] = interp_meta
            # Check interpolation validity
            if not interp_meta.get('valid', True) or wl_interp is None or ref_interp is None:
                reason = f"interpolation_failed: {interp_meta.get('message', 'invalid interpolation')}"
                self.logger.warning(f"Interpolation failed for {filepath}: {reason}")
                self.error_log.append(f"{filepath}: {reason}")
                self._record_rejection(
                    file_path=str(filepath),
                    library=source_library,
                    stage='interpolation',
                    reason=reason,
                    n_points=int(np.sum(~np.isnan(wl_clean))) if wl_clean is not None else 0,
                    wavelength_min=float(np.nanmin(wl_clean)) if wl_clean is not None and wl_clean.size>0 else None,
                    wavelength_max=float(np.nanmax(wl_clean)) if wl_clean is not None and wl_clean.size>0 else None,
                    missing_count=int(np.sum(np.isnan(ref_clean))) if ref_clean is not None else None,
                    mineral=mineral_name
                )
                return False, None, None
            
            # Update wavelength statistics
            self.wavelength_stats['min_wavelength'] = min(
                self.wavelength_stats['min_wavelength'], 
                float(np.nanmin(wl_interp))
            )
            self.wavelength_stats['max_wavelength'] = max(
                self.wavelength_stats['max_wavelength'], 
                float(np.nanmax(wl_interp))
            )
            self.wavelength_stats['coverage_values'].append(interp_meta['coverage'])
            
            # STAGE 2b: Standardization - Normalization
            ref_norm, norm_meta = normalize_spectrum(ref_interp, method=self.config.normalization_method)
            
            # Validate length consistency after normalization
            if wl_interp is not None and ref_norm is not None:
                if len(wl_interp) != len(ref_norm):
                    self.logger.error(f"Length mismatch after normalization: wavelengths={len(wl_interp)}, reflectance={len(ref_norm)}")
                    self._record_rejection(
                        file_path=str(filepath),
                        library=source_library,
                        stage='normalization_length_validation',
                        reason=f'Length mismatch after normalization: {len(wl_interp)} vs {len(ref_norm)}',
                        n_points=len(wl_interp),
                        mineral=mineral_name
                    )
                    return False, None, None
            
            processed_data['reflectance_norm'] = ref_norm
            processing_stages['normalization'] = norm_meta
            
            # Save standardized spectrum
            std_dir = Path(self.config.output_paths['standardized']) / mineral_name
            std_dir.mkdir(parents=True, exist_ok=True)
            self._save_spectrum(wl_interp, ref_norm, std_dir / f"{spectrum_id}_standardized.parquet")
            
            # STAGE 3: Continuum Removal
            ref_cr, ref_continuum, cr_meta = remove_continuum(wl_interp, ref_norm, method='convex_hull')
            
            # Validate length consistency after continuum removal
            if wl_interp is not None and ref_cr is not None:
                if len(wl_interp) != len(ref_cr):
                    self.logger.error(f"Length mismatch after continuum removal: wavelengths={len(wl_interp)}, reflectance={len(ref_cr)}")
                    self._record_rejection(
                        file_path=str(filepath),
                        library=source_library,
                        stage='continuum_removal_length_validation',
                        reason=f'Length mismatch after continuum removal: {len(wl_interp)} vs {len(ref_cr)}',
                        n_points=len(wl_interp),
                        mineral=mineral_name
                    )
                    return False, None, None
            
            processed_data['reflectance_continuum_removed'] = ref_cr
            processing_stages['continuum_removal'] = cr_meta
            # Check continuum removal validity
            if not cr_meta.get('valid', True) or ref_cr is None or np.all(np.isnan(ref_cr)):
                reason = f"continuum_removal_failed: {cr_meta.get('message', 'invalid continuum') }"
                self.logger.warning(f"Continuum removal failed for {filepath}: {reason}")
                self.error_log.append(f"{filepath}: {reason}")
                self._record_rejection(
                    file_path=str(filepath),
                    library=source_library,
                    stage='continuum_removal',
                    reason=reason,
                    n_points=int(np.sum(~np.isnan(wl_interp))) if wl_interp is not None else 0,
                    wavelength_min=float(np.nanmin(wl_interp)) if wl_interp is not None and wl_interp.size>0 else None,
                    wavelength_max=float(np.nanmax(wl_interp)) if wl_interp is not None and wl_interp.size>0 else None,
                    missing_count=int(np.sum(np.isnan(ref_norm))) if ref_norm is not None else None,
                    mineral=mineral_name
                )
                return False, None, None
            
            # Save continuum-removed spectrum
            cr_dir = Path(self.config.output_paths['continuum_removed']) / mineral_name
            cr_dir.mkdir(parents=True, exist_ok=True)
            self._save_spectrum(wl_interp, ref_cr, cr_dir / f"{spectrum_id}_continuum_removed.parquet")
            
            # STAGE 3.5: Derivative Spectra
            derivatives = compute_derivatives(ref_norm, wl_interp, orders=[1, 2])
            
            # Save derivative spectra
            deriv_dir = Path(self.config.output_paths['derivatives']) / mineral_name
            deriv_dir.mkdir(parents=True, exist_ok=True)
            
            if 1 in derivatives:
                deriv_1, meta_1 = derivatives[1]
                self._save_spectrum(wl_interp, deriv_1, deriv_dir / f"{spectrum_id}_derivative_1.parquet")
                processing_stages['derivative_1'] = meta_1
            
            if 2 in derivatives:
                deriv_2, meta_2 = derivatives[2]
                self._save_spectrum(wl_interp, deriv_2, deriv_dir / f"{spectrum_id}_derivative_2.parquet")
                processing_stages['derivative_2'] = meta_2
            
            # Create metadata record
            metadata_record = create_spectrum_metadata(
                spectrum_id=spectrum_id,
                mineral_name=mineral_name,
                source_library=source_library,
                original_filename=filepath.name,
                raw_wavelengths=wl_raw,
                cleaned_wavelengths=wl_clean,
                processing_stages=processing_stages,
                measurement_type='reflectance',
                instrument='unknown',
                collection_date='unknown',
                grain_size='unknown',
                sample_origin='unknown',
                # pass diagnostics
                coverage=interp_meta.get('coverage') if interp_meta else None,
                interpolated_points=interp_meta.get('output_points') if interp_meta else None,
                pre_variance=cleaning_report.get('pre_variance'),
                post_variance=cleaning_report.get('post_variance'),
                negatives_clipped=cleaning_report.get('negatives_clipped', 0),
                outliers_adjusted=cleaning_report.get('outliers_adjusted', 0)
            )

            return True, metadata_record, processed_data
        
        except Exception as e:
            error_msg = f"Error processing {filepath}: {str(e)}"
            self.logger.error(error_msg)
            self.error_log.append(error_msg)
            # record rejection with exception
            self._record_rejection(
                file_path=str(filepath),
                library=source_library,
                stage='processing_exception',
                reason=str(e),
                n_points=0,
                wavelength_min=None,
                wavelength_max=None,
                missing_count=None,
                mineral=mineral_name
            )
            return False, None, None
    
    def _save_spectrum(self, wavelengths: np.ndarray, reflectance: np.ndarray, filepath: Path) -> None:
        """Save spectrum to Parquet file."""
        df = pd.DataFrame({
            'wavelength': wavelengths,
            'reflectance': reflectance
        })
        df.to_parquet(filepath, index=False)
    
    def run(self, output_summary_file: Path = None) -> None:
        """Run the complete preprocessing pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("SPECTRAL DATABASE PREPROCESSING PIPELINE")
        self.logger.info("=" * 80)
        
        # Discover spectra
        self.logger.info("\n[1/8] Discovering spectra...")
        spectra_by_library = self.discover_spectra()
        
        total_spectra = sum(len(specs) for lib in spectra_by_library.values() 
                           for specs in lib.values())
        
        # Process all spectra
        self.logger.info(f"\n[2-7/8] Processing {total_spectra} spectra...")
        
        with tqdm(total=total_spectra, desc="Processing spectra") as pbar:
            for library, minerals_dict in spectra_by_library.items():
                for mineral_name, filepaths in minerals_dict.items():
                    for filepath in filepaths:
                        success, metadata, _ = self.process_spectrum(
                            filepath, mineral_name, library
                        )
                        
                        self.processing_stats['total_processed'] += 1
                        self.processing_stats['library_processed'][library] = (
                            self.processing_stats['library_processed'].get(library, 0) + 1
                        )

                        if success:
                            self.processing_stats['total_accepted'] += 1
                            self.processing_stats['library_accepted'][library] = (
                                self.processing_stats['library_accepted'].get(library, 0) + 1
                            )
                            self.metadata_records.append(metadata)

                            # Update statistics
                            if mineral_name not in self.processing_stats['mineral_counts']:
                                self.processing_stats['mineral_counts'][mineral_name] = 0
                            self.processing_stats['mineral_counts'][mineral_name] += 1

                            if library not in self.processing_stats['library_counts']:
                                self.processing_stats['library_counts'][library] = 0
                            self.processing_stats['library_counts'][library] += 1
                        else:
                            self.processing_stats['total_rejected'] += 1
                            self.processing_stats['library_rejected'][library] = (
                                self.processing_stats['library_rejected'].get(library, 0) + 1
                            )

                        pbar.update(1)

        relab_stats = {
            'discovered': self.processing_stats['library_discovered'].get('relab', 0),
            'processed': self.processing_stats['library_processed'].get('relab', 0),
            'accepted': self.processing_stats['library_accepted'].get('relab', 0),
            'rejected': self.processing_stats['library_rejected'].get('relab', 0),
        }
        self.logger.info(
            "RELAB summary: discovered=%s, processed=%s, accepted=%s, rejected=%s",
            relab_stats['discovered'],
            relab_stats['processed'],
            relab_stats['accepted'],
            relab_stats['rejected'],
        )
        print(
            f"RELAB summary: discovered={relab_stats['discovered']}, "
            f"processed={relab_stats['processed']}, accepted={relab_stats['accepted']}, "
            f"rejected={relab_stats['rejected']}"
        )
        
        # Build metadata database
        self.logger.info("\n[8/8] Building metadata database and generating reports...")
        
        metadata_path = Path(self.config.output_paths['outputs']) / 'metadata_database.csv'
        if self.metadata_records:
            metadata_df = build_metadata_database(self.metadata_records, metadata_path)

            # Generate summary statistics
            summary_path = Path(self.config.output_paths['outputs']) / 'summary_statistics.csv'
            generate_summary_statistics(metadata_df, summary_path)
        else:
            self.logger.warning("No spectra were successfully processed")
            metadata_df = pd.DataFrame()
        
        # Generate report
        processing_params = {
            'savgol_window': self.config.savgol_window,
            'savgol_polyorder': self.config.savgol_polyorder,
            'interpolation_method': self.config.interpolation_method,
            'wavelength_min': self.config.wavelength_min,
            'wavelength_max': self.config.wavelength_max,
            'wavelength_step': self.config.wavelength_step,
            'normalization_method': self.config.normalization_method,
            'max_missing_fraction': self.config.max_missing_fraction,
            'outlier_threshold': self.config.outlier_threshold,
            'min_spectrum_points': self.config.min_spectrum_points,
            'integrity_metrics': {
                'avg_coverage_per_mineral': None,
                'total_negatives_clipped': int(self.cleaning_stats.get('total_negatives_clipped',0)),
                'total_outliers_adjusted': int(self.cleaning_stats.get('total_outliers_adjusted',0)),
                'avg_variance_ratio': None
            }
        }

        if not metadata_df.empty:
            try:
                if 'coverage' in metadata_df.columns:
                    avg_cov = metadata_df.groupby('mineral_name')['coverage'].mean()
                    processing_params['integrity_metrics']['avg_coverage_per_mineral'] = float(avg_cov.mean())

                if 'pre_variance' in metadata_df.columns and 'post_variance' in metadata_df.columns:
                    ratios = metadata_df.apply(
                        lambda r: (r['post_variance'] / r['pre_variance'])
                        if r['pre_variance'] and r['pre_variance'] > 0 else None,
                        axis=1,
                    )
                    ratios = [x for x in ratios if x is not None]
                    if ratios:
                        processing_params['integrity_metrics']['avg_variance_ratio'] = float(sum(ratios) / len(ratios))
            except Exception as e:
                self.logger.warning(f"Failed to compute integrity metrics: {e}")
        
        wavelength_stats_report = {
            'min_wavelength': self.wavelength_stats['min_wavelength'],
            'max_wavelength': self.wavelength_stats['max_wavelength'],
            'wavelength_range': self.wavelength_stats['max_wavelength'] - self.wavelength_stats['min_wavelength'],
            'mean_coverage': np.mean(self.wavelength_stats['coverage_values']) if self.wavelength_stats['coverage_values'] else 0.0,
        }
        
        report_path = Path(self.config.output_paths['outputs']) / 'preprocessing_report.md'
        # Write rejection report CSV
        rejection_path = Path(self.config.output_paths['outputs']) / 'rejection_report.csv'
        if self.rejection_records:
            try:
                rej_df = pd.DataFrame(self.rejection_records)
                rej_df.to_csv(rejection_path, index=False, encoding='utf-8')
                self.logger.info(f"Rejection report written: {rejection_path}")
            except Exception as e:
                self.logger.error(f"Failed to write rejection report: {e}")
        else:
            self.logger.info("No rejections to write to report.")

        generate_markdown_report(
            report_path,
            self.processing_stats,
            self.processing_stats['mineral_counts'],
            self.processing_stats['library_counts'],
            wavelength_stats_report,
            self.cleaning_stats,
            processing_params,
            self.error_log,
            rejection_records=self.rejection_records
        )
        
        # Print summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PREPROCESSING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total processed: {self.processing_stats['total_processed']}")
        self.logger.info(f"Total accepted: {self.processing_stats['total_accepted']}")
        self.logger.info(f"Total rejected: {self.processing_stats['total_rejected']}")
        self.logger.info(f"Acceptance rate: {(self.processing_stats['total_accepted']/max(1, self.processing_stats['total_processed']))*100:.1f}%")
        self.logger.info("\nOutput files:")
        self.logger.info(f"  - Metadata: {metadata_path}")
        self.logger.info(f"  - Summary: {Path(self.config.output_paths['outputs']) / 'summary_statistics.csv'}")
        self.logger.info(f"  - Report: {report_path}")
        self.logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    preprocessor = SpectralPreprocessor()
    preprocessor.run()
