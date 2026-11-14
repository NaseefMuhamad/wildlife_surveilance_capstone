"""Compatibility inference entry point; accepts the same arguments as src/detect.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pipeline_cli import predict_main

if __name__ == '__main__':
    predict_main(default_source='data/raw/TestReal/TestReal/images/image.png')
