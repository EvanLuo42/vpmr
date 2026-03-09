#!/usr/bin/env -S uv run
"""Convert STL (or other mesh formats) to PLY for VPMR pipeline input.

Usage:
    uv run scripts/convert_stl.py <input_mesh> <output_dir>

Converts the input mesh to PLY format and saves it to <output_dir>/<stem>.ply.
"""

import argparse
import sys
from pathlib import Path

import trimesh


def main():
    parser = argparse.ArgumentParser(description="Convert mesh to PLY for VPMR")
    parser.add_argument("input_mesh", help="Path to input mesh (STL, OBJ, OFF, etc.)")
    parser.add_argument("output_dir", help="Output directory")
    args = parser.parse_args()

    src = Path(args.input_mesh)
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(src), process=False, force="mesh")
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}.ply"
    mesh.export(str(dst))
    print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
