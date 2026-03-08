#!/usr/bin/env python3
"""
Download 100 models from the Thingi10K dataset suitable for testing VPMR.

Selects models with defects (non-watertight, self-intersecting, non-manifold, etc.)
that are good candidates for mesh repair while filtering out overly large/complex ones.

Usage:
    uv run python scripts/download_thingi10k.py [--output-dir data/thingi10k] [--count 100]
"""

import argparse
import json
import shutil
from pathlib import Path

import thingi10k


def select_and_download(output_dir: Path, count: int = 100):
    """Select models suitable for VPMR testing and copy them to output_dir."""

    print("Initializing Thingi10K dataset (downloading if needed)...")
    thingi10k.init(variant="raw")

    # Collect models by defect category, filtering by face count
    face_range = (100, 200_000)

    categories = {
        "non-manifold": dict(manifold=False, num_facets=face_range),
        "self-intersecting": dict(self_intersecting=True, manifold=True, closed=False, num_facets=face_range),
        "non-watertight": dict(closed=False, manifold=True, self_intersecting=False, num_facets=face_range),
        "multi-component": dict(num_components=(2, None), solid=True, num_facets=face_range),
        "non-oriented": dict(oriented=False, num_facets=face_range),
        "clean": dict(solid=True, manifold=True, num_components=1, num_facets=face_range),
    }

    quotas = {
        "non-manifold": 25,
        "self-intersecting": 20,
        "non-watertight": 25,
        "multi-component": 10,
        "non-oriented": 10,
        "clean": 10,
    }

    selected = []
    selected_ids = set()

    for cat_name, filters in categories.items():
        quota = quotas[cat_name]
        ds = thingi10k.dataset(**filters)
        print(f"  {cat_name}: {len(ds)} available, selecting {quota}")

        # Sort by face count distance from 20K (prefer medium-sized)
        entries = list(ds)
        entries.sort(key=lambda e: abs(e["num_facets"] - 20000))

        taken = 0
        for entry in entries:
            if taken >= quota:
                break
            fid = entry["file_id"]
            if fid in selected_ids:
                continue
            selected.append((entry, cat_name))
            selected_ids.add(fid)
            taken += 1

    # Fill remaining slots from defective models
    remaining = count - len(selected)
    if remaining > 0:
        for cat_name in ["non-manifold", "self-intersecting", "non-watertight"]:
            ds = thingi10k.dataset(**categories[cat_name])
            for entry in ds:
                if remaining <= 0:
                    break
                fid = entry["file_id"]
                if fid not in selected_ids:
                    selected.append((entry, cat_name))
                    selected_ids.add(fid)
                    remaining -= 1

    selected = selected[:count]
    print(f"\nTotal selected: {len(selected)} models")

    # Copy files and build manifest
    print(f"Copying to {output_dir}/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, (entry, cat_name) in enumerate(selected):
        fid = entry["file_id"]
        src = Path(entry["file_path"])
        dst = output_dir / f"{fid}{src.suffix}"

        if not dst.exists():
            shutil.copy2(src, dst)

        print(f"  [{i+1}/{len(selected)}] {fid} ({entry.get('name', '?')}, "
              f"{entry['num_facets']} faces, {cat_name})")

        manifest.append({
            "file_id": fid,
            "file": dst.name,
            "name": entry.get("name", ""),
            "category": cat_name,
            "num_vertices": entry.get("num_vertices"),
            "num_facets": entry.get("num_facets"),
            "num_components": entry.get("num_components"),
            "closed": entry.get("closed"),
            "manifold": entry.get("manifold"),
            "self_intersecting": entry.get("self_intersecting"),
            "oriented": entry.get("oriented"),
            "solid": entry.get("solid"),
            "genus": entry.get("genus"),
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} models saved to {output_dir}/")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Thingi10K models for VPMR testing")
    parser.add_argument("--output-dir", type=str, default="data/thingi10k",
                        help="Output directory (default: data/thingi10k)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of models to download (default: 100)")
    args = parser.parse_args()

    select_and_download(Path(args.output_dir), args.count)


if __name__ == "__main__":
    main()
