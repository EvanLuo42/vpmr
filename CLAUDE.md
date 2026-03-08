# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Implementation of the paper "Visual-Preserving Mesh Repair" (VPMR). Converts defective triangle meshes into watertight manifold meshes using ray-tracing-based visual measures to guide repair while preserving visual appearance and UV attributes.

## Build

Requires: CMake 3.23+, Ninja, Clang (or GCC), CUDA toolkit, OptiX SDK, Eigen3, CGAL, GMP/MPFR.

```bash
# Configure (OptiX path is required)
cmake --preset linux-clang-release -DOPTIX_INSTALL_DIR=/home/phakel/.local/share/optix

# Build
cmake --build build/linux-clang-release

# Debug build
cmake --preset linux-clang-debug -DOPTIX_INSTALL_DIR=/home/phakel/.local/share/optix
cmake --build build/linux-clang-debug
```

GCC presets also available: `linux-gcc-release`, `linux-gcc-debug`.

## Run

```bash
./build/linux-clang-release/vpmr <input_mesh.ply> <output_mesh.ply>
```

Test mesh: `data/bunny/reconstruction/bun_zipper.ply`

## Architecture

The pipeline runs sequentially through 8 stages in `source/vpmr.cpp:run_pipeline()`, matching paper sections:

1. **Visual Measures** (`source/visual_measures/`) — §3.1: Computes per-face visibility, orientation, and openness via GPU ray tracing. `OptixRayTracer` uses OptiX with multi-bounce tracing in a single kernel launch. CUDA kernel in `optix_programs.cu`, compiled to PTX at build time via `cmake/embed_ptx.cmake`.

2. **Orientation** (`source/orientation/`) — §3.2: Flood-fills consistently-oriented patches, flips patches with negative area-weighted orientation measure.

3. **Offset** (`source/offset/`) — §3.3: For open surfaces (high boundary ratio), creates offset shell by duplicating open faces with reversed winding and stitching sides. For nearly-closed meshes (boundary_ratio < 5%), does simple hole-fill instead.

4. **Partition** (`source/partition/`) — §3.4: Space partition using BSP tree (`bsp.cpp`) built on CGAL Delaunay tetrahedralization with exact arithmetic (CGAL `Exact_predicates_exact_constructions_kernel`). Input mesh first processed through libigl `remesh_self_intersections` to resolve intersections. Faces inserted into BSP, then cells extracted. `face_mapping` tracks which partition faces correspond to original input faces via three passes: BSP labels, vertex-triple hash, barycenter containment.

5. **Graph Cut** (`source/graph_cut/`) — §3.5: Min-cut/max-flow on cell adjacency graph using external `maxflow` library. Cell 0 = unbounded/exterior (forced SINK). Visible faces provide data term (Eq. 7), invisible/extra faces provide smoothness term (Eq. 8). Interface faces between interior/exterior cells form the output.

6. **Simplification** (`source/simplification/`) — §3.6: Merges coplanar patches via constrained ear-clipping. Respects geometric boundary edges and UV seams. Includes self-intersection avoidance check within extended bounding box.

7. **Topology** (`source/topology/`) — §3.7: Fixes non-manifold edges (>2 incident faces) and non-manifold vertices (disconnected face fans) by vertex duplication.

8. **Attributes** (`source/attributes/`) — §3.8: Recovers UV coordinates via barycentric interpolation from original faces (using `face_mapping` and `offset_source`), flood-fills remaining faces.

### Key Data Structures

- **`Mesh`** (`include/vpmr/mesh.h`): Core struct with `V` (vertices), `F` (faces), per-face `visibility`/`orientation`/`openness`, `face_mapping` (partition face → original face), `offset_source` (offset face → source face), and UV data (`TC`, `FTC`, `VC`).
- **`PartitionResult`** (`source/partition/partition.h`): Partition mesh + `per_patch_cells` (per-face cell adjacency), `source_visibility`/`source_orientation` propagated from pre-partition mesh.
- **`Config`** (`include/vpmr/config.h`): Pipeline parameters (sampling counts, thresholds, offset divisor).
- **`BSPTree`** (`source/partition/bsp.h`): Exact-arithmetic BSP built on CGAL Delaunay tets.

### External Dependencies (in `external/`)

- **libigl**: Mesh processing (self-intersection resolution, `extract_cells`, `boundary_loop`, `remove_unreferenced`)
- **maxflow**: Boykov-Kolmogorov max-flow/min-cut solver

### Conventions

- Eigen type aliases in `include/vpmr/types.h`: `MatXd`, `MatXi`, `VecXd`, `VecXi`, `Vec3d`, etc.
- `face_normal()` uses right-hand rule: `(v1-v0).cross(v2-v0)`
- `per_patch_cells(f,0)` = positive cell (normal side), `per_patch_cells(f,1)` = negative cell
- Cell 0 = infinite/unbounded cell (exterior)
- In maxflow: SOURCE = Interior, SINK = Exterior
- `face_mapping` values: >= 0 means original input face index, < 0 means extra face (created by partition/BSP)
- CGAL 6.x compat: `CGAL_Core` removed, CMakeLists.txt has alias shim
