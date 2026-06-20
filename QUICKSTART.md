# Quick Start Guide

## 1. Setup (First Time)

```bash
# Install dependencies
pip install -r requirements.txt

# Create sample data directory structure (for testing)
mkdir -p data/raw/RELAB/Quartz
mkdir -p data/raw/RRUFF/Calcite
mkdir -p data/raw/USGS/Olivine
```

## 2. Prepare Input Data

Place your spectrum files in the appropriate directories:
```
data/raw/RELAB/Mineral_Name/spectrum.csv
data/raw/RRUFF/Mineral_Name/spectrum.txt
data/raw/USGS/Mineral_Name/spectrum.asc
```

Each file should have two columns: wavelength and reflectance.

## 3. Configure Settings (Optional)

Edit `config/settings.yaml` to customize processing:
- Wavelength range and resolution
- Smoothing parameters
- Normalization method
- Quality control thresholds

## 4. Run Pipeline

```bash
# Run with default settings
python main.py

# Or with custom config
python main.py --config config/settings.yaml
```

## 5. Check Results

Processing outputs are saved in:
- `outputs/metadata_database.csv` - Metadata for all processed spectra
- `outputs/preprocessing_report.md` - Detailed processing report
- `outputs/summary_statistics.csv` - Statistical summary
- `logs/preprocessing.log` - Execution log

Processed spectra are organized by stage:
- `data/cleaned/<mineral>/` - Cleaned spectra
- `data/standardized/<mineral>/` - Interpolated and normalized spectra
- `data/continuum_removed/<mineral>/` - Continuum-removed spectra
- `data/derivatives/<mineral>/` - First and second derivatives

## 6. Example Spectrum File

Create a test file `data/raw/USGS/Quartz/quartz_sample.csv`:

```
wavelength,reflectance
400,0.045
401,0.046
402,0.047
403,0.048
404,0.049
405,0.050
...
2500,0.120
```

Then run:
```bash
python main.py
```

## Key Features

- **Automatic discovery**: Finds all spectra in data/raw/ directories
- **Quality control**: Validates spectra before processing
- **Modular design**: Each stage can be run independently
- **Complete logging**: All operations logged to preprocessing.log
- **Metadata preservation**: All processing steps tracked

## Troubleshooting

**No spectra found?**
- Ensure spectra are in `data/raw/LIBRARY/MINERAL/` directories
- Check file extensions (`.csv`, `.txt`, or `.asc`)

**Processing errors?**
- Check `logs/preprocessing.log` for details
- Verify spectrum file format (two columns)
- Ensure numeric wavelength and reflectance values

**Performance?**
- For large datasets, processing may take several minutes
- Memory usage is typically < 1 GB
- Progress bar shows estimated time remaining

## Output Examples

### Metadata Database (metadata_database.csv)
```
spectrum_id,mineral_name,source_library,wavelength_min,wavelength_max,num_points,quality_flag
quartz_s1_abc12345,Quartz,USGS,400.0,2500.0,2101,OK
calcite_s2_def67890,Calcite,RRUFF,400.0,2500.0,2101,OK
```

### Report (preprocessing_report.md)
- Dataset summary with acceptance rate
- Per-mineral and per-library statistics
- Wavelength coverage analysis
- Cleaning statistics
- Processing parameters
- Error log summary

## Next Steps

After preprocessing:

1. **Load processed data** into analysis tools:
```python
import pandas as pd
df = pd.read_parquet('data/standardized/Quartz/spectrum_id_standardized.parquet')
```

2. **Analyze metadata**:
```python
metadata = pd.read_csv('outputs/metadata_database.csv')
print(metadata.groupby('mineral_name').size())
```

3. **Use derivatives** for feature analysis
4. **Apply machine learning** on standardized spectra

---

For detailed documentation, see `README.md`
