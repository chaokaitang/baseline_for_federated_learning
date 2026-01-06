"""
Robust EMNIST downloader (ZIP source)

Downloads the provided `gzip.zip` from the NIST biometrics site, extracts only
the EMNIST balanced files (those containing 'balanced' in their name) and places
the resulting .gz/.idx files in `data/EMNIST/raw/` under the script directory.

Usage:
    python download_emnist.py

Notes:
 - The script streams the download, shows progress, retries on transient errors
   and extracts only the matching balanced files to avoid unnecessary extras.
 - If automatic download fails, follow the printed manual instructions.
"""

import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


# URL provided by user
ZIP_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'

RETRY = 3
CHUNK_SIZE = 1024 * 1024

cpath = Path(__file__).parent
DATA_DIR = cpath / 'data'
RAW_DIR = DATA_DIR / 'EMNIST' / 'raw'
ZIP_PATH = DATA_DIR / 'gzip.zip'


def download_with_progress(url, dest_path, retries=RETRY):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                total = resp.getheader('Content-Length')
                total = int(total) if total is not None else None
                downloaded = 0
                start = time.time()
                with open(dest_path, 'wb') as out:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100.0 / total
                            elapsed = time.time() - start
                            speed = downloaded / 1024 / elapsed if elapsed > 0 else 0
                            sys.stdout.write(f"\rDownloaded {downloaded}/{total} bytes ({percent:.1f}%) - {speed:.1f} KB/s")
                        else:
                            sys.stdout.write(f"\rDownloaded {downloaded} bytes")
                        sys.stdout.flush()
                sys.stdout.write('\n')
            return True
        except Exception as e:
            print(f"Attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    return False


# def extract_balanced_from_zip(zip_path, dest_raw_dir):
#     """Extract files from zip that contain 'balanced' in their name and end with .gz or .ubyte"""
#     extracted = []
#     try:
#         with zipfile.ZipFile(zip_path, 'r') as zf:
#             for member in zf.namelist():
#                 lower = member.lower()
#                 if 'balanced' in lower and (lower.endswith('.gz') or lower.endswith('.ubyte')):
#                     # Extract to temporary location (DATA_DIR) then move files to RAW_DIR
#                     print(f"Extracting {member}...")
#                     try:
#                         zf.extract(member, path=DATA_DIR)
#                         src = DATA_DIR / member
#                         # Ensure destination directory exists
#                         dest = dest_raw_dir / os.path.basename(member)
#                         dest_raw_dir.mkdir(parents=True, exist_ok=True)
#                         # Move/rename
#                         if src.exists():
#                             src.replace(dest)
#                         else:
#                             # Some zips have flattened names
#                             alt = DATA_DIR / os.path.basename(member)
#                             if alt.exists():
#                                 alt.replace(dest)
#                         extracted.append(dest)
#                     except Exception as ex:
#                         print(f"  Failed to extract {member}: {ex}")
#     except Exception as e:
#         print(f"Error opening zip file: {e}")
#     return extracted


def main():
    print('=' * 80)
    print('EMNIST-balanced ZIP downloader')
    print('=' * 80)

    print(f"Target raw directory: {RAW_DIR}")
    print(f"ZIP url: {ZIP_URL}")

    if ZIP_PATH.exists():
        print(f"Zip file already exists at {ZIP_PATH}, skipping download.")
    else:
        print('\nStarting download...')
        ok = download_with_progress(ZIP_URL, ZIP_PATH)
        if not ok:
            print('\nDownload failed after retries.')
            print('\nPlease download manually from:')
            print(ZIP_URL)
            print(f'And extract the balanced files into: {RAW_DIR}')
            return

    print('\nDownload finished.')
    print(f'ZIP saved to: {ZIP_PATH}')
    print('Note: this downloader does NOT extract files. Run generate_emnist_niid.py which will extract needed balanced files from the ZIP when required.')
    print('\nDone.')


if __name__ == '__main__':
    main()
