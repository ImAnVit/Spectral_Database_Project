"""
Configuration manager for loading and validating settings.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Load, validate, and provide access to configuration settings."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    @property
    def wavelength_min(self) -> float:
        return self.get('wavelength.min')
    
    @property
    def wavelength_max(self) -> float:
        return self.get('wavelength.max')
    
    @property
    def wavelength_step(self) -> float:
        return self.get('wavelength.step')
    
    @property
    def savgol_window(self) -> int:
        return self.get('smoothing.window_length')
    
    @property
    def savgol_polyorder(self) -> int:
        return self.get('smoothing.polyorder')
    
    @property
    def normalization_method(self) -> str:
        return self.get('processing.normalization_method')
    
    @property
    def interpolation_method(self) -> str:
        return self.get('processing.interpolation_method')
    
    @property
    def max_missing_fraction(self) -> float:
        return self.get('quality_control.max_missing_fraction')
    
    @property
    def min_spectrum_points(self) -> int:
        return self.get('quality_control.min_spectrum_points')
    
    @property
    def max_noise_level(self) -> float:
        return self.get('quality_control.max_noise_level')
    
    @property
    def outlier_threshold(self) -> float:
        return self.get('quality_control.outlier_threshold')
    
    @property
    def detect_outliers(self) -> bool:
        return self.get('quality_control.detect_outliers')
    
    @property
    def raw_data_paths(self) -> Dict[str, str]:
        return {
            'relab': self.get('data_paths.raw_relab'),
            'rruff': self.get('data_paths.raw_rruff'),
            'usgs': self.get('data_paths.raw_usgs'),
        }
    
    @property
    def output_paths(self) -> Dict[str, str]:
        return {
            'cleaned': self.get('data_paths.cleaned'),
            'standardized': self.get('data_paths.standardized'),
            'continuum_removed': self.get('data_paths.continuum_removed'),
            'derivatives': self.get('data_paths.derivatives'),
            'metadata': self.get('data_paths.metadata'),
            'outputs': self.get('data_paths.outputs'),
        }
    
    @property
    def minerals(self) -> list:
        return self.get('minerals', [])
    
    @property
    def log_level(self) -> str:
        return self.get('logging.level')
    
    @property
    def log_file(self) -> str:
        return self.get('logging.log_file')
    
    @property
    def log_format(self) -> str:
        return self.get('logging.log_format')
