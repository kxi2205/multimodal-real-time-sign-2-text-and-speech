"""
Downloader for external model files.

Usage:
  - Host your model files somewhere public or authenticated (S3, GDrive, etc).
  - Update the MODELS list with direct download URLs and filenames.
  - Run: python scripts/download_models.py

This script downloads models into `backend/models/` and will create the directory if it doesn't exist.
"""

import os
import sys
from pathlib import Path

try:
    # Python 3
    from urllib.request import urlretrieve
except Exception:
    from urllib import urlretrieve

MODELS = [
    # ('https://example.com/path/to/model_1.pkl', 'model_1.pkl'),
    # ('https://example.com/path/to/scaler_1.pkl', 'scaler_1.pkl'),
]

DEST_DIR = Path(__file__).resolve().parent.parent / 'backend' / 'models'


def download(url, dest_path):
    print(f"Downloading {url} -> {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, str(dest_path))
        print("Done")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def main():
    if not MODELS:
        print("No model URLs configured. Edit this file and add URLs to MODELS.")
        sys.exit(1)

    for url, fname in MODELS:
        dest = DEST_DIR / fname
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        download(url, dest)

    print("All downloads complete.")


if __name__ == '__main__':
    main()
