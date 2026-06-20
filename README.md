# Spectral Database Preprocessing Pipeline

A research-grade Python project for preprocessing mineral reflectance spectra from RELAB, RRUFF, and USGS spectral libraries. Designed for machine learning studies in spectral feature analysis, crystal symmetry research, spectral complexity analysis, and symbolic regression.

## Features

### Fully Automated Pipeline
- **Spectrum Discovery**: Automatically discovers and catalogs spectra from multiple libraries
- **Quality Control**: Comprehensive validation with configurable thresholds
- **Modular Architecture**: Each processing stage can be configured and modified independently
- **Complete Logging**: Detailed tracking of all operations and issues
- **Progress Tracking**: Real-time progress bars with tqdm

### Processing Stages

1. **Cleaning & Quality Control**
   - NaN removal and interpolation
   - Duplicate wavelength detection
   - Negative reflectance value removal
   - Outlier detection (z-score method)
   - Savitzky-Golay smoothing (configurable window and polynomial order)
   - Wavelength ordering validation

2. **Standardization**
   - Interpolation onto common wavelength grid (400-2500 nm, 1 nm spacing)
   - Multiple normalization methods:
     - Min-Max (default)
     - Z-score standardization
     - Max normalization
     - Robust scaling

3. **Continuum Removal**
   - Convex hull-based continuum estimation
   - Absorption feature enhancement
   - Spectral normalization

4. **Derivative Spectra**
   - First derivative (dR/dλ)
   - Second derivative (d²R/dλ²)
   - Finite difference calculation
   - Useful for absorption band analysis

5. **Metadata Database**
   - Comprehensive spectrum metadata tracking
   - Processing history preservation
   - Quality flags and statistics
   - CSV export for downstream analysis

6. **Reporting**
   - Markdown preprocessing report
   - Summary statistics
   - Per-mineral and per-library statistics
   - Wavelength coverage analysis
   - Error logging

## Project Structure

```
spectral_database_project/
├── config/
│   └── settings.yaml              # Configuration file
├── data/
│   ├── raw/
│   │   ├── RELAB/
│   │   ├── RRUFF/
│   │   └── USGS/
│   ├── cleaned/                   # Cleaning output
│   ├── standardized/              # Interpolation + normalization output
│   ├── continuum_removed/         # Continuum removal output
│   ├── derivatives/               # Derivative spectra output
│   └── metadata/
├── logs/
│   └── preprocessing.log
├── notebooks/                     # Jupyter notebooks for analysis
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── utils.py                   # Logging and utilities
│   ├── cleaning/
│   │   ├── clean_spectra.py       # Cleaning pipeline
│   │   └── validate_spectra.py    # Quality control
│   ├── standardization/
│   │   ├── interpolate.py         # Wavelength interpolation
│   │   └── normalize.py           # Spectrum normalization
│   ├── continuum/
│   │   └── continuum_removal.py   # Continuum estimation & removal
│   ├── derivatives/
│   │   └── derivative_features.py # Derivative calculation
│   ├── metadata/
│   │   └── build_metadata.py      # Metadata database creation
│   ├── reporting/
│   │   └── generate_report.py     # Report generation
│   └── pipeline/
│       └── preprocess_all.py      # Master pipeline orchestrator
├── outputs/
│   ├── cleaned_dataset.parquet    # (example)
│   ├── metadata_database.csv
│   ├── preprocessing_report.md
│   └── summary_statistics.csv
├── requirements.txt               # Python dependencies
├── main.py                        # Entry point
└── README.md                      # This file
```

## Installation

### 1. Clone or Download the Project

```bash
cd spectral_database_project
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml` to customize processing parameters:

```yaml
# Wavelength range and resolution
wavelength:
  min: 400        # Minimum wavelength (nm)
  max: 2500       # Maximum wavelength (nm)
  step: 1         # Interpolation step (nm)

# Savitzky-Goyal smoothing
smoothing:
  window_length: 11    # Filter window (must be odd)
  polyorder: 3         # Polynomial order

# Normalization and interpolation
processing:
  normalization_method: "minmax"  # Options: minmax, stddev, max, robust_scale
  interpolation_method: "linear"  # Options: linear, cubic, quadratic

# Quality control thresholds
quality_control:
  max_missing_fraction: 0.20      # Allow up to 20% missing values
  min_spectrum_points: 50         # Minimum valid points
  max_noise_level: 0.05           # Maximum acceptable noise
  detect_outliers: true
  outlier_threshold: 3.0          # Standard deviations

# Logging level
logging:
  level: "INFO"   # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Usage

### Run with Default Configuration

```bash
python main.py
```

### Run with Custom Configuration

```bash
python main.py --config config/custom.yaml
```

### Programmatic Usage

```python
from src.pipeline.preprocess_all import SpectralPreprocessor

# Initialize preprocessor
preprocessor = SpectralPreprocessor(config_path="config/settings.yaml")

# Run pipeline
preprocessor.run()
```

## Input Data Format

Raw spectra should be organized as:

```
data/raw/RELAB/Mineral_Name/spectrum1.csv
data/raw/RRUFF/Mineral_Name/spectrum2.txt
data/raw/USGS/Mineral_Name/spectrum3.asc
```

Supported file formats:
- **CSV** with comma, semicolon, or tab delimiters
- **TXT** with whitespace delimiters
- **ASCII** text files

Spectrum files must contain two columns:
1. Wavelength (nm)
2. Reflectance (0-1 or 0-100)

Example CSV format:
```
wavelength,reflectance
400.0,0.045
401.0,0.046
402.0,0.047
...
```

## Output Products

The pipeline generates:

### 1. Processed Spectra (Parquet Format)

- `data/cleaned/<mineral>/<spectrum_id>_cleaned.parquet`
- `data/standardized/<mineral>/<spectrum_id>_standardized.parquet`
- `data/continuum_removed/<mineral>/<spectrum_id>_continuum_removed.parquet`
- `data/derivatives/<mineral>/<spectrum_id>_derivative_1.parquet` (first derivative)
- `data/derivatives/<mineral>/<spectrum_id>_derivative_2.parquet` (second derivative)

### 2. Metadata Database

`outputs/metadata_database.csv` - Contains:
- spectrum_id
- mineral_name
- source_library
- original_filename
- measurement_type
- instrument
- wavelength_min, wavelength_max
- num_points
- processing_steps
- quality_flag

### 3. Summary Statistics

`outputs/summary_statistics.csv` - Contains:
- Total spectra processed/accepted/rejected
- Per-mineral counts
- Per-library counts
- Wavelength statistics

### 4. Preprocessing Report

`outputs/preprocessing_report.md` - Detailed markdown report with:
- Dataset summary
- Mineral and library statistics
- Wavelength coverage analysis
- Cleaning statistics
- Processing parameters used
- Error log summary

### 5. Log File

`logs/preprocessing.log` - Complete execution log with timestamps

## Supported Minerals

The pipeline is configured for these 20 minerals:

Grossular, Pyrope, Spinel, Andradite, Augite, Diopside, Muscovite, Orthoclase, Enstatite, Olivine, Hypersthene, Zircon, Rutile, Albite, Microcline, Apatite, Calcite, Dolomite, Quartz, Beryl

Modify the `minerals` list in `config/settings.yaml` to include additional minerals.

## Advanced Features

### Custom Processing Parameters

Modify processing stages by editing configuration:

```yaml
smoothing:
  window_length: 21    # Stronger smoothing
  polyorder: 4         # Higher order polynomial

quality_control:
  outlier_threshold: 2.0  # Stricter outlier detection
```

### Parallel Processing

Enable multi-core processing (future enhancement):

```yaml
processing_options:
  n_jobs: -1  # Use all available cores
```

## Troubleshooting

### No Spectra Discovered

Check that:
1. Raw data paths exist in `data/raw/`
2. Spectra are organized by library (RELAB/RRUFF/USGS) and mineral
3. Files have `.csv`, `.txt`, or `.asc` extension

### Processing Errors

1. Check `logs/preprocessing.log` for detailed error messages
2. Verify spectrum file format (two columns: wavelength, reflectance)
3. Ensure wavelength and reflectance values are numeric
4. Check for corrupted or malformed files

### Low Acceptance Rate

Adjust quality control thresholds in `config/settings.yaml`:

```yaml
quality_control:
  max_missing_fraction: 0.30   # Increase tolerance
  outlier_threshold: 4.0        # Relax outlier detection
  min_spectrum_points: 30       # Lower minimum points
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation and CSV export
- **scipy**: Interpolation and convex hull
- **scikit-learn**: Scaling and preprocessing
- **pyarrow**: Parquet file support
- **PyYAML**: Configuration file parsing
- **tqdm**: Progress bars
- **click**: CLI support (future)

See `requirements.txt` for specific versions.

## Performance Notes

- Processing time depends on spectrum count and size
- Typical processing: 1000 spectra in 5-10 minutes
- Memory usage: ~100 MB for typical datasets
- Output file sizes:
  - Cleaned spectrum (~2 KB per Parquet file)
  - Metadata database (~100 KB for 1000 spectra)
  - Report and statistics (~50 KB total)

## Quality Metrics

The pipeline computes:

- **Coverage**: Fraction of output wavelength grid with valid data
- **Noise estimate**: Local variance within moving window
- **Outlier count**: Number of points exceeding threshold
- **Data loss**: NaNs, duplicates, negatives removed

## Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex
@software{spectral_preprocessing_2026,
  title={Spectral Database Preprocessing Pipeline},
  author={AnVit},
  year={2026},
  url={https://github.com/ImAnVit/spectral_database_project}
}
```

## License

This project is provided as-is for research purposes.

## Contributing

For bug reports, feature requests, or contributions, please open an issue or pull request.

## Future Enhancements

- [ ] Parallel spectrum processing
- [ ] Interactive visualization notebooks
- [ ] Machine learning feature extraction
- [ ] Symbolic regression analysis
- [ ] Web-based preprocessing interface
- [ ] Uncertainty quantification
- [ ] Spectral feature database

## Contact & Support

For questions or technical support, contact the project maintainer.

---

**Last Updated**: 2026
**Version**: 1.0.0
