"""
Report generation module for preprocessing results.
"""
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime
import numpy as np


logger = logging.getLogger(__name__)


def generate_markdown_report(
    output_path: Path,
    processing_summary: Dict,
    mineral_stats: Dict,
    library_stats: Dict,
    wavelength_stats: Dict,
    cleaning_stats: Dict,
    processing_params: Dict,
    error_log: List[str],
    rejection_records: List[Dict] = None
) -> None:
    """
    Generate comprehensive preprocessing report in Markdown format.
    
    Args:
        output_path: Path to save report
        processing_summary: Summary of processing statistics
        mineral_stats: Per-mineral statistics
        library_stats: Per-library statistics
        wavelength_stats: Wavelength coverage statistics
        cleaning_stats: Cleaning operation statistics
        processing_params: Processing parameters used
        error_log: List of errors/warnings encountered
        rejection_records: Optional list of rejection records for diagnostics
    """
    
    report_lines = []
    
    # Header
    report_lines.append("# Spectral Database Preprocessing Report")
    report_lines.append(f"\nGenerated: {datetime.now().isoformat()}\n")
    
    # Dataset Summary
    report_lines.append("## Dataset Summary\n")
    report_lines.append(f"- **Total spectra processed**: {processing_summary.get('total_processed', 0)}")
    report_lines.append(f"- **Total spectra accepted**: {processing_summary.get('total_accepted', 0)}")
    report_lines.append(f"- **Total spectra rejected**: {processing_summary.get('total_rejected', 0)}")
    acceptance_rate = 0.0
    if processing_summary.get('total_processed', 0) > 0:
        acceptance_rate = (processing_summary.get('total_accepted', 0) / processing_summary.get('total_processed', 0)) * 100
    report_lines.append(f"- **Acceptance rate**: {acceptance_rate:.1f}%\n")
    
    # Mineral Statistics
    report_lines.append("## Mineral Statistics\n")
    report_lines.append("| Mineral | Count |")
    report_lines.append("|---------|-------|")
    for mineral, count in sorted(mineral_stats.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"| {mineral} | {count} |")
    report_lines.append("")
    
    # Library Statistics
    report_lines.append("## Source Library Statistics\n")
    report_lines.append("| Library | Count |")
    report_lines.append("|---------|-------|")
    for library, count in sorted(library_stats.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"| {library} | {count} |")
    report_lines.append("")
    
    # Wavelength Statistics
    report_lines.append("## Wavelength Statistics\n")
    report_lines.append(f"- **Minimum wavelength**: {wavelength_stats.get('min_wavelength', 'N/A')} nm")
    report_lines.append(f"- **Maximum wavelength**: {wavelength_stats.get('max_wavelength', 'N/A')} nm")
    mean_coverage = wavelength_stats.get('mean_coverage')
    mean_coverage_str = f"{mean_coverage:.1%}" if isinstance(mean_coverage, (int, float)) else "N/A"
    report_lines.append(f"- **Mean coverage**: {mean_coverage_str}")
    report_lines.append(f"- **Wavelength range**: {wavelength_stats.get('wavelength_range', 'N/A')} nm\n")
    # Integrity metrics (if provided)
    integrity = processing_params.get('integrity_metrics', {}) if processing_params else {}
    if integrity:
        report_lines.append("## Spectral Integrity Metrics\n")
        avg_coverage = integrity.get('avg_coverage_per_mineral')
        avg_coverage_str = f"{avg_coverage:.1%}" if isinstance(avg_coverage, (int, float)) else "N/A"
        report_lines.append(f"- **Average coverage per mineral**: {avg_coverage_str}")
        report_lines.append(f"- **Total negatives clipped**: {integrity.get('total_negatives_clipped', 0)}")
        report_lines.append(f"- **Total outlier adjustments**: {integrity.get('total_outliers_adjusted', 0)}")
        report_lines.append(f"- **Average pre/post variance ratio**: {integrity.get('avg_variance_ratio', 'N/A')}")
        report_lines.append("")
    
    # Cleaning Statistics
    report_lines.append("## Cleaning Statistics\n")
    report_lines.append(f"- **NaNs removed**: {cleaning_stats.get('total_nans_removed', 0)}")
    report_lines.append(f"- **Duplicates removed**: {cleaning_stats.get('total_duplicates_removed', 0)}")
    report_lines.append(f"- **Outlier adjustments**: {cleaning_stats.get('total_outliers_adjusted', 0)}")
    report_lines.append(f"- **Negatives clipped**: {cleaning_stats.get('total_negatives_clipped', 0)}")
    report_lines.append(f"- **Spectra rejected (cleaning)**: {cleaning_stats.get('cleaning_failures', 0)}")
    report_lines.append(f"- **Significantly altered spectra**: {cleaning_stats.get('significantly_altered_count', 0)}\n")
    
    # Per-library cleaning statistics
    library_cleaning_stats = cleaning_stats.get('library_cleaning_stats', {})
    if library_cleaning_stats:
        report_lines.append("### Per-Library Cleaning Intensity\n")
        report_lines.append("| Library | Processed | NaNs Removed | Duplicates Removed | Outliers Adjusted | Negatives Clipped |")
        report_lines.append("|---------|-----------|-------------|-------------------|-------------------|-------------------|")
        for lib in ['relab', 'rruff', 'usgs']:
            if lib in library_cleaning_stats and library_cleaning_stats[lib]:
                stats = library_cleaning_stats[lib]
                report_lines.append(f"| {lib.upper()} | {stats.get('processed_count', 0)} | {stats.get('nans_removed', 0)} | {stats.get('duplicates_removed', 0)} | {stats.get('outliers_adjusted', 0)} | {stats.get('negatives_clipped', 0)} |")
        report_lines.append("")
    
    # Spectral distortion statistics
    distortion_stats = cleaning_stats.get('spectral_distortion_stats', [])
    if distortion_stats:
        report_lines.append("## Spectral Distortion Analysis\n")
        
        # Compute statistics
        sam_angles = [d.get('sam_angle', 0) for d in distortion_stats if d.get('sam_angle') is not None]
        cosine_sims = [d.get('cosine_similarity', 0) for d in distortion_stats if d.get('cosine_similarity') is not None]
        distortion_scores = [d.get('distortion_score', 0) for d in distortion_stats if d.get('distortion_score') is not None]
        variance_ratios = [d.get('variance_ratio', 0) for d in distortion_stats if d.get('variance_ratio') is not None]
        
        if sam_angles:
            report_lines.append("### Spectral Angle Mapper (SAM) Distribution\n")
            report_lines.append(f"- **Mean SAM angle**: {np.mean(sam_angles):.4f} radians")
            report_lines.append(f"- **Median SAM angle**: {np.median(sam_angles):.4f} radians")
            report_lines.append(f"- **Min SAM angle**: {np.min(sam_angles):.4f} radians")
            report_lines.append(f"- **Max SAM angle**: {np.max(sam_angles):.4f} radians")
            report_lines.append(f"- **Std SAM angle**: {np.std(sam_angles):.4f} radians\n")
        
        if cosine_sims:
            report_lines.append("### Cosine Similarity Distribution\n")
            report_lines.append(f"- **Mean cosine similarity**: {np.mean(cosine_sims):.4f}")
            report_lines.append(f"- **Median cosine similarity**: {np.median(cosine_sims):.4f}")
            report_lines.append(f"- **Min cosine similarity**: {np.min(cosine_sims):.4f}")
            report_lines.append(f"- **Max cosine similarity**: {np.max(cosine_sims):.4f}\n")
        
        if distortion_scores:
            report_lines.append("### Combined Distortion Score Distribution\n")
            report_lines.append(f"- **Mean distortion score**: {np.mean(distortion_scores):.4f}")
            report_lines.append(f"- **Median distortion score**: {np.median(distortion_scores):.4f}")
            report_lines.append(f"- **Min distortion score**: {np.min(distortion_scores):.4f}")
            report_lines.append(f"- **Max distortion score**: {np.max(distortion_scores):.4f}")
            report_lines.append(f"- **Spectra with distortion > 0.15**: {sum(1 for s in distortion_scores if s > 0.15)}\n")
        
        if variance_ratios:
            report_lines.append("### Variance Preservation Ratio Distribution\n")
            report_lines.append(f"- **Mean variance ratio**: {np.mean(variance_ratios):.4f}")
            report_lines.append(f"- **Median variance ratio**: {np.median(variance_ratios):.4f}")
            report_lines.append(f"- **Min variance ratio**: {np.min(variance_ratios):.4f}")
            report_lines.append(f"- **Max variance ratio**: {np.max(variance_ratios):.4f}")
            report_lines.append(f"- **Over-smoothed (ratio < 0.5)**: {sum(1 for r in variance_ratios if r < 0.5)}")
            report_lines.append(f"- **Over-amplified (ratio > 2.0)**: {sum(1 for r in variance_ratios if r > 2.0)}\n")
        
        # Per-library distortion analysis
        report_lines.append("### Per-Library Distortion Analysis\n")
        for lib in ['relab', 'rruff', 'usgs']:
            lib_distortion = [d for d in distortion_stats if d.get('library') == lib]
            if lib_distortion:
                lib_sam = [d.get('sam_angle', 0) for d in lib_distortion if d.get('sam_angle') is not None]
                lib_dist_score = [d.get('distortion_score', 0) for d in lib_distortion if d.get('distortion_score') is not None]
                if lib_sam:
                    report_lines.append(f"**{lib.upper()}**:")
                    report_lines.append(f"- Mean SAM angle: {np.mean(lib_sam):.4f} radians")
                    report_lines.append(f"- Mean distortion score: {np.mean(lib_dist_score):.4f}" if lib_dist_score else "- Mean distortion score: N/A")
                    report_lines.append("")
    
    # Processing Parameters
    report_lines.append("## Processing Parameters\n")
    report_lines.append("### Smoothing")
    report_lines.append(f"- Window length: {processing_params.get('savgol_window', 'N/A')}")
    report_lines.append(f"- Polynomial order: {processing_params.get('savgol_polyorder', 'N/A')}")
    
    report_lines.append("\n### Interpolation")
    report_lines.append(f"- Method: {processing_params.get('interpolation_method', 'N/A')}")
    report_lines.append(f"- Wavelength range: {processing_params.get('wavelength_min', 'N/A')}-{processing_params.get('wavelength_max', 'N/A')} nm")
    report_lines.append(f"- Step size: {processing_params.get('wavelength_step', 'N/A')} nm")
    
    report_lines.append("\n### Normalization")
    report_lines.append(f"- Method: {processing_params.get('normalization_method', 'N/A')}\n")
    
    # Quality Control
    report_lines.append("## Quality Control Parameters\n")
    max_missing = processing_params.get('max_missing_fraction')
    max_missing_str = f"{max_missing:.1%}" if isinstance(max_missing, (int, float)) else "N/A"
    report_lines.append(f"- Max missing fraction: {max_missing_str}")
    report_lines.append(f"- Outlier threshold: {processing_params.get('outlier_threshold', 'N/A')} σ")
    report_lines.append(f"- Minimum spectrum points: {processing_params.get('min_spectrum_points', 'N/A')}\n")
    
    # Error Log Summary
    if error_log:
        report_lines.append("## Error Log Summary\n")
        report_lines.append(f"Total issues reported: {len(error_log)}\n")
        report_lines.append("### Recent Issues")
        for error in error_log[-20:]:  # Show last 20 errors
            report_lines.append(f"- {error}")
        report_lines.append("")
    else:
        report_lines.append("## Error Log Summary\n")
        report_lines.append("No errors or warnings reported.\n")

    # Rejection diagnostics
    if rejection_records:
        report_lines.append("## Rejection Diagnostics\n")
        report_lines.append(f"- Total rejections: {len(rejection_records)}\n")
        from collections import Counter
        stages = Counter([r.get('stage','unknown') for r in rejection_records])
        report_lines.append("### Rejections by Stage")
        for stage, count in stages.most_common():
            report_lines.append(f"- {stage}: {count}")
        report_lines.append("")
        libs = Counter([r.get('library','unknown') for r in rejection_records])
        report_lines.append("### Rejections by Library")
        for lib, count in libs.most_common():
            report_lines.append(f"- {lib}: {count}")
        report_lines.append("")
        reasons = Counter([r.get('reason','') for r in rejection_records])
        report_lines.append("### Top Failure Reasons")
        for reason, count in reasons.most_common(10):
            report_lines.append(f"- {reason}: {count}")
        report_lines.append("")
        # Acceptance rate per mineral/library
        report_lines.append("### Acceptance Rates\n")
        # acceptance by mineral
        report_lines.append("#### Acceptance rate per mineral")
        total_per_mineral = Counter()
        accepted_per_mineral = Counter()
        for m, c in mineral_stats.items():
            total_per_mineral[m] = c
            accepted_per_mineral[m] = c
        # rejections may include mineral field
        for r in rejection_records:
            m = r.get('mineral','unknown')
            total_per_mineral[m] += 1
            accepted_per_mineral[m] += 0
        for mineral in sorted(total_per_mineral.keys()):
            tot = total_per_mineral[mineral]
            acc = accepted_per_mineral.get(mineral, 0)
            rate = (acc / tot * 100) if tot > 0 else 0.0
            report_lines.append(f"- {mineral}: {acc}/{tot} ({rate:.1f}%)")
        report_lines.append("")
        # acceptance by library
        report_lines.append("#### Acceptance rate per library")
        total_per_lib = Counter(library_stats)
        accepted_per_lib = Counter(library_stats)
        for r in rejection_records:
            lib = r.get('library','unknown')
            total_per_lib[lib] += 1
            accepted_per_lib[lib] += 0
        for lib in total_per_lib:
            tot = total_per_lib[lib]
            acc = accepted_per_lib.get(lib, 0)
            rate = (acc / tot * 100) if tot > 0 else 0.0
            report_lines.append(f"- {lib}: {acc}/{tot} ({rate:.1f}%)")
        report_lines.append("")
    else:
        report_lines.append("## Rejection Diagnostics\n")
        report_lines.append("No rejections recorded.\n")

    # Footer
    report_lines.append("---\n")
    report_lines.append("*Generated by Spectral Database Preprocessing Pipeline*")
    
    # Write report
    report_text = '\n'.join(report_lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Report generated: {output_path}")
