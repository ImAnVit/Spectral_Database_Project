"""
Test/validation script for the preprocessing pipeline.

This script verifies that all components are correctly installed and working.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...", end=" ")
    try:
        import numpy
        import pandas
        import scipy
        import sklearn
        import yaml
        import pyarrow
        import tqdm
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("Testing configuration...", end=" ")
    try:
        from src.config import ConfigManager
        config = ConfigManager("config/settings.yaml")
        
        assert config.wavelength_min == 400
        assert config.wavelength_max == 2500
        assert config.savgol_window == 11
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_cleaning():
    """Test cleaning module."""
    print("Testing cleaning module...", end=" ")
    try:
        import numpy as np
        from src.cleaning.clean_spectra import clean_spectrum
        
        # Create synthetic spectrum
        wl = np.linspace(400, 2500, 100)
        ref = 0.05 + 0.02 * np.sin(wl / 200) + np.random.normal(0, 0.001, len(wl))
        
        wl_clean, ref_clean, report = clean_spectrum(wl, ref)
        
        assert report['valid'], f"Cleaning failed: {report}"
        assert len(wl_clean) > 0
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_standardization():
    """Test standardization module."""
    print("Testing standardization...", end=" ")
    try:
        import numpy as np
        from src.standardization.interpolate import interpolate_spectrum
        from src.standardization.normalize import normalize_spectrum
        
        # Create synthetic spectrum
        wl = np.linspace(400, 2500, 50)
        ref = 0.05 + 0.02 * np.sin(wl / 200)
        
        # Test interpolation
        wl_interp, ref_interp, meta = interpolate_spectrum(wl, ref)
        assert len(wl_interp) > len(wl), "Interpolation failed"
        
        # Test normalization
        ref_norm, norm_meta = normalize_spectrum(ref_interp, method='minmax')
        assert np.nanmax(ref_norm) <= 1.0, "Normalization failed"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_continuum():
    """Test continuum removal."""
    print("Testing continuum removal...", end=" ")
    try:
        import numpy as np
        from src.continuum.continuum_removal import remove_continuum
        
        # Create synthetic spectrum
        wl = np.linspace(400, 2500, 100)
        ref = 0.1 + 0.05 * np.sin(wl / 200) - 0.02 * ((wl - 1200) ** 2) / 1e6
        ref = np.clip(ref, 0, 1)
        
        ref_cr, ref_cont, stats = remove_continuum(wl, ref)
        
        assert stats['valid'], f"Continuum removal failed: {stats}"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_derivatives():
    """Test derivative calculation."""
    print("Testing derivatives...", end=" ")
    try:
        import numpy as np
        from src.derivatives.derivative_features import compute_derivatives
        
        # Create synthetic spectrum
        wl = np.linspace(400, 2500, 100)
        ref = 0.05 + 0.02 * np.sin(wl / 200)
        
        derivatives = compute_derivatives(ref, wl, orders=[1, 2])
        
        assert 1 in derivatives, "First derivative not computed"
        assert 2 in derivatives, "Second derivative not computed"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("Testing directory structure...", end=" ")
    try:
        required_dirs = [
            'config',
            'data/raw',
            'data/cleaned',
            'data/standardized',
            'data/continuum_removed',
            'data/derivatives',
            'logs',
            'outputs',
            'src',
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                print(f"✗ Missing directory: {dir_path}")
                return False
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SPECTRAL PREPROCESSING PIPELINE - VALIDATION TEST")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_directory_structure,
        test_config,
        test_cleaning,
        test_standardization,
        test_continuum,
        test_derivatives,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 60)
        print("\nPipeline is ready to use!")
        print("Run: python main.py")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total})")
        print("=" * 60)
        print("\nPlease fix the issues and try again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
