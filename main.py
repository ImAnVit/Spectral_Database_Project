#!/usr/bin/env python3
"""
Main entry point for Spectral Database Preprocessing Pipeline.

Usage:
    python main.py              # Run with default config
    python main.py --config config/custom.yaml
"""
import argparse
import sys
from pathlib import Path

from src.pipeline.preprocess_all import SpectralPreprocessor


def main():
    """Parse arguments and run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Spectral Database Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                         # Use default config/settings.yaml
  python main.py --config config/custom.yaml
        '''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Run preprocessing
    try:
        preprocessor = SpectralPreprocessor(config_path=args.config)
        preprocessor.run()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
