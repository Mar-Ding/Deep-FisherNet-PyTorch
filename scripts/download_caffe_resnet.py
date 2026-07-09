"""
download_caffe_resnet.py — Download standard Caffe ResNet-101 model for backbone alignment.

Multiple download sources (tried in order):
  1. OneDrive (original KaimingHe ResNet-101) - may require cookies
  2. Google Drive (soeaver/caffe-model resnet101-v2) - fallback
  3. Baidu Pan (soeaver/caffe-model) - China-friendly

Usage:
  python scripts/download_caffe_resnet.py --out-dir external/official-fishernet/models/Res-101/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


# Direct download URLs for ResNet-101 Caffe model
# Primary: from Kaiming He's original OneDrive
# Fallback: from soeaver's caffe-model repo (resnet101-v2, may differ from v1)
SOURCES = [
    {
        "name": "OneDrive (KaimingHe, ResNet-101 v1)",
        "url": "https://onedrive.live.com/download?resid=4006CBB8476FF777%2117887&authkey=%21AAFW2-FVoxeVRck",
        "filename": "ResNet-101-model.caffemodel",
    },
    {
        "name": "Google Drive (soeaver, resnet101-v2)",
        "url": "https://drive.google.com/uc?export=download&id=0B9mkjlmP0d7zRlhISks0VktGOGs",
        "filename": "resnet101_v2.caffemodel",
    },
]


def try_download(source: dict, out_dir: Path, timeout: int = 120) -> Path | None:
    """Try a single source; return path if successful, None otherwise."""
    url = source["url"]
    filename = source["filename"]
    name = source["name"]
    out_path = out_dir / filename

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Already exists: {out_path} ({size_mb:.1f} MB)")
        return out_path

    print(f"\n  Trying [{name}]")
    print(f"  URL: {url}")
    print(f"  → {out_path}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        if total < 10 * 1024 * 1024:  # less than 10MB => probably an error page
            print(f"  Content too small ({total / 1024:.1f} KB), likely a redirect/error page. Skip.")
            return None

        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="\r")
        print(f"\n  ✓ Downloaded: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return out_path

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        # Clean up partial download
        if out_path.exists():
            out_path.unlink()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Caffe ResNet-101 model.")
    parser.add_argument("--out-dir", type=str, default="external/official-fishernet/models/Res-101/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        result = try_download(source, out_dir)
        if result is not None:
            print(f"\n✓ Successfully downloaded: {result}")
            return

    # If we get here, all sources failed
    print("\n" + "=" * 60)
    print("All download sources failed. Please download manually:")
    print("  1. KaimingHe OneDrive (original ResNet-101 v1):")
    print("     https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777")
    print("     → Download ResNet-101-model.caffemodel")
    print("     → Save to: " + str(out_dir / "ResNet-101-model.caffemodel"))
    print("  2. Or try: https://github.com/soeaver/caffe-model")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
