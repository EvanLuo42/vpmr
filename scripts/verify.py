#!/usr/bin/env -S uv run
"""Verify mesh repair output: mesh info, watertight/manifold checks, HD/LFD/PSNR metrics.

Metrics follow the VPMR paper (Zheng et al.):
- HD: Hausdorff distance, normalized by bounding box diagonal
- LFD: silhouette descriptor distance from 20 fixed views
- PSNR: Peak signal-to-noise ratio on rendered images
  - Input rendered with double-face rendering (both sides visible) as reference
  - Output rendered with front faces white and back faces black

Usage:
    uv run scripts/verify.py <input_mesh> <output_mesh>
"""

import argparse
import math
import sys
import warnings
from collections import defaultdict, deque

import numpy as np

warnings.filterwarnings("ignore")


def load_mesh(path: str):
    import trimesh

    mesh = trimesh.load(path, process=False, force="mesh")
    # Weld coincident vertices so topology checks are meaningful
    # (STL triangle soup has 3 unique verts per face with no sharing)
    mesh.merge_vertices()
    return mesh


def mesh_info(mesh, label: str):
    print(f"  {label}:")
    print(f"    Vertices: {len(mesh.vertices)}")
    print(f"    Faces:    {len(mesh.faces)}")


def is_watertight(mesh) -> bool:
    """Every edge has exactly 2 incident faces."""
    return bool(mesh.is_watertight)


def _count_boundary_loops(boundary_edges):
    boundary_adjacency = defaultdict(set)
    for u, v in boundary_edges:
        boundary_adjacency[u].add(v)
        boundary_adjacency[v].add(u)

    visited = set()
    closed_loops = 0
    non_loop_components = 0

    for start in boundary_adjacency:
        if start in visited:
            continue

        component = set()
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            if vertex in visited:
                continue
            visited.add(vertex)
            component.add(vertex)
            queue.extend(boundary_adjacency[vertex] - visited)

        if all(len(boundary_adjacency[vertex]) == 2 for vertex in component):
            closed_loops += 1
        else:
            non_loop_components += 1

    return closed_loops, non_loop_components, len(boundary_adjacency)


def topology_diagnostics(mesh):
    """Collect topology diagnostics used by the verifier output."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_vertices = len(mesh.vertices)
    stats = {
        "manifold": False,
        "degenerate_faces": 0,
        "duplicate_faces": 0,
        "non_manifold_edges": 0,
        "non_manifold_vertices": 0,
        "unreferenced_vertices": 0,
        "boundary_edges": 0,
        "boundary_loops": 0,
        "boundary_non_loop_components": 0,
        "boundary_vertices": 0,
    }

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        return stats

    if np.any(faces < 0) or np.any(faces >= n_vertices):
        return stats

    # Repeated indices inside a face are always invalid.
    degenerate_mask = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    stats["degenerate_faces"] = int(np.count_nonzero(degenerate_mask))
    if stats["degenerate_faces"] > 0:
        return stats

    # Coincident duplicate faces are not manifold even if edge counts stay <= 2.
    sorted_faces = np.sort(faces, axis=1)
    unique_faces = {tuple(face) for face in sorted_faces}
    stats["duplicate_faces"] = len(faces) - len(unique_faces)
    if stats["duplicate_faces"] > 0:
        return stats

    edge_to_faces = defaultdict(list)
    vertex_to_faces = [[] for _ in range(n_vertices)]
    referenced = np.zeros(n_vertices, dtype=bool)

    for face_index, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for vertex in (a, b, c):
            vertex_to_faces[vertex].append(face_index)
            referenced[vertex] = True
        for edge in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted(edge))].append(face_index)

    stats["unreferenced_vertices"] = int(np.count_nonzero(~referenced))
    boundary_edges = [edge for edge, adjacent_faces in edge_to_faces.items() if len(adjacent_faces) == 1]
    stats["boundary_edges"] = len(boundary_edges)
    (
        stats["boundary_loops"],
        stats["boundary_non_loop_components"],
        stats["boundary_vertices"],
    ) = _count_boundary_loops(boundary_edges)

    stats["non_manifold_edges"] = sum(
        len(adjacent_faces) > 2 for adjacent_faces in edge_to_faces.values()
    )

    for vertex, incident_faces in enumerate(vertex_to_faces):
        if not incident_faces:
            continue

        local_adjacency = {face_index: set() for face_index in incident_faces}
        for face_index in incident_faces:
            face = faces[face_index]
            neighbors = [int(v) for v in face if int(v) != vertex]
            for neighbor in neighbors:
                edge = tuple(sorted((vertex, neighbor)))
                for adjacent_face in edge_to_faces[edge]:
                    if adjacent_face != face_index:
                        local_adjacency[face_index].add(adjacent_face)

        invalid = False
        if any(len(face_neighbors) > 2 for face_neighbors in local_adjacency.values()):
            invalid = True

        visited = set()
        queue = deque([incident_faces[0]])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(local_adjacency[current] - visited)

        if len(visited) != len(incident_faces):
            invalid = True

        degrees = [len(local_adjacency[face_index]) for face_index in incident_faces]
        if len(incident_faces) == 1:
            if degrees[0] != 0:
                invalid = True
        else:
            degree_one_count = sum(degree == 1 for degree in degrees)
            degree_zero_count = sum(degree == 0 for degree in degrees)
            if degree_zero_count != 0:
                invalid = True
            if degree_one_count not in (0, 2):
                invalid = True

        if invalid:
            stats["non_manifold_vertices"] += 1

    stats["manifold"] = (
        stats["degenerate_faces"] == 0
        and stats["duplicate_faces"] == 0
        and stats["non_manifold_edges"] == 0
        and stats["non_manifold_vertices"] == 0
        and stats["unreferenced_vertices"] == 0
    )
    return stats


def is_manifold(mesh) -> bool:
    """Check edge-manifoldness and vertex fan connectivity."""
    return topology_diagnostics(mesh)["manifold"]


def hausdorff_distance(mesh_a, mesh_b, n_samples: int = 100000, seed: int = 0) -> float:
    """Symmetric Hausdorff distance normalized by bounding box diagonal of mesh_a."""
    import trimesh
    from scipy.spatial import cKDTree

    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples, seed=seed)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples, seed=seed + 1)

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a)
    dist_b2a, _ = tree_a.query(pts_b)

    hd = max(dist_a2b.max(), dist_b2a.max())

    # Normalize by bounding box diagonal of input
    bbox_diag = np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0])
    if bbox_diag > 0:
        hd /= bbox_diag

    return round(hd, 4)


# --- Rendering utilities for LFD and PSNR ---

def _camera_positions(n_views: int = 20):
    """Generate camera positions on a sphere."""
    if n_views == 20:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi
        positions = []

        for x in (-1.0, 1.0):
            for y in (-1.0, 1.0):
                for z in (-1.0, 1.0):
                    positions.append(np.array([x, y, z], dtype=np.float64))

        for y in (-inv_phi, inv_phi):
            for z in (-phi, phi):
                positions.append(np.array([0.0, y, z], dtype=np.float64))

        for x in (-inv_phi, inv_phi):
            for y in (-phi, phi):
                positions.append(np.array([x, y, 0.0], dtype=np.float64))

        for x in (-phi, phi):
            for z in (-inv_phi, inv_phi):
                positions.append(np.array([x, 0.0, z], dtype=np.float64))

        return [position / np.linalg.norm(position) for position in positions]

    positions = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_views):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_views)
        phi = 2 * np.pi * i / golden_ratio
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        positions.append(np.array([x, y, z]))
    return positions


def _shared_render_frame(*meshes):
    bounds = np.array([mesh.bounds for mesh in meshes], dtype=np.float64)
    lower = bounds[:, 0, :].min(axis=0)
    upper = bounds[:, 1, :].max(axis=0)
    center = 0.5 * (lower + upper)
    scale = np.max(upper - lower)
    return center, scale if scale > 0 else 1.0


def _look_at(eye, target, up=np.array([0, 1, 0])):
    """Compute a 4x4 camera-to-world matrix looking from eye to target."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([1, 0, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def _render_views(mesh, camera_positions, center, scale, resolution=256, render_mode="reference"):
    """Render a mesh from multiple viewpoints into grayscale images."""
    import pyrender
    import trimesh as tm

    verts = (mesh.vertices - center) / scale
    images = []
    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    try:
        for cam_pos in camera_positions:
            scene = pyrender.Scene(bg_color=[0, 0, 0, 255], ambient_light=[0.2, 0.2, 0.2])
            _add_render_meshes(scene, tm, pyrender, verts, mesh.faces, render_mode)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
            camera_pose = _look_at(cam_pos * 2.5, np.zeros(3))

            # Directional light from camera position for shading
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(light, pose=camera_pose)
            scene.add(camera, pose=camera_pose)

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            images.append(np.mean(color[:, :, :3].astype(np.float64), axis=2))
    finally:
        renderer.delete()

    return images


def _material(pyrender, rgb):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=False,
    )


def _add_render_meshes(scene, tm, pyrender, vertices, faces, render_mode):
    front_mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
    back_mesh = tm.Trimesh(vertices=vertices, faces=faces[:, ::-1], process=False)

    white = _material(pyrender, (1.0, 1.0, 1.0))
    black = _material(pyrender, (0.0, 0.0, 0.0))

    scene.add(pyrender.Mesh.from_trimesh(front_mesh, material=white, smooth=False))
    if render_mode == "front_only":
        pass  # front faces only, no backface rendering
    elif render_mode == "reference":
        scene.add(pyrender.Mesh.from_trimesh(back_mesh, material=white, smooth=False))
    elif render_mode == "backface_black":
        scene.add(pyrender.Mesh.from_trimesh(back_mesh, material=black, smooth=False))
    else:
        raise ValueError(f"Unknown render mode: {render_mode}")


def compute_lfd(images_a, images_b) -> float:
    """Compute a silhouette descriptor distance across all views."""
    max_order = 12
    total_dist = 0.0

    for img_a, img_b in zip(images_a, images_b):
        sil_a = (img_a > 1).astype(np.float64)
        sil_b = (img_b > 1).astype(np.float64)
        desc_a = _zernike_moments(sil_a, max_order)
        desc_b = _zernike_moments(sil_b, max_order)
        total_dist += np.linalg.norm(desc_a - desc_b)

    return round(total_dist, 1)


def _zernike_radial(n: int, m: int, rho):
    """Compute radial polynomial R_nm(rho) for Zernike moments."""
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        num = ((-1) ** s) * math.factorial(n - s)
        den = (
            math.factorial(s)
            * math.factorial((n + m) // 2 - s)
            * math.factorial((n - m) // 2 - s)
        )
        R += (num / den) * (rho ** (n - 2 * s))
    return R


def _zernike_moments(image, max_order: int = 12):
    """Compute Zernike moment magnitudes up to given order.

    Maps image to unit disk, computes |Z_nm| for all valid (n, m>=0) pairs.
    Returns a 1D array of moment magnitudes (the descriptor).
    """
    h, w = image.shape
    # Map pixel coordinates to unit disk
    y, x = np.mgrid[0:h, 0:w]
    x = (2.0 * x - w + 1) / (w - 1)
    y = (2.0 * y - h + 1) / (h - 1)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    mask = rho <= 1.0

    n_pixels = np.sum(mask)
    if n_pixels == 0:
        # Count expected moments
        count = sum(1 for n in range(max_order + 1) for m in range(n % 2, n + 1, 2))
        return np.zeros(count)

    moments = []
    for n in range(max_order + 1):
        for m in range(n % 2, n + 1, 2):
            R = _zernike_radial(n, m, rho)
            basis = R * np.exp(-1j * m * theta)
            Z = np.sum(image[mask] * np.conj(basis[mask])) * (n + 1) / (n_pixels)
            moments.append(abs(Z))

    return np.array(moments)


def compute_psnr(images_a, images_b) -> float:
    """Average PSNR across all rendered views.

    PSNR = 10 * log10(MAX^2 / MSE), where MAX = 255 for 8-bit images.
    """
    psnr_values = []
    for img_a, img_b in zip(images_a, images_b):
        mse = np.mean((img_a - img_b) ** 2)
        if mse < 1e-10:
            psnr_values.append(100.0)  # cap at 100 dB for identical images
        else:
            psnr = 10.0 * np.log10(255.0**2 / mse)
            psnr_values.append(psnr)

    return round(np.mean(psnr_values), 1)


def print_topology_diagnostics(stats, label: str):
    print(f"  {label} diagnostics:")
    print(f"    Boundary edges:              {stats['boundary_edges']}")
    print(f"    Boundary loops:              {stats['boundary_loops']}")
    print(f"    Boundary non-loop comps:     {stats['boundary_non_loop_components']}")
    print(f"    Boundary vertices:           {stats['boundary_vertices']}")
    print(f"    Non-manifold edges:          {stats['non_manifold_edges']}")
    print(f"    Non-manifold vertices:       {stats['non_manifold_vertices']}")
    print(f"    Unreferenced vertices:       {stats['unreferenced_vertices']}")
    print(f"    Duplicate faces:             {stats['duplicate_faces']}")
    print(f"    Degenerate faces:            {stats['degenerate_faces']}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify mesh repair: mesh info, watertight/manifold, HD/LFD/PSNR."
    )
    parser.add_argument("input_mesh", help="Path to input mesh")
    parser.add_argument("output_mesh", help="Path to output (repaired) mesh")
    parser.add_argument("--samples", type=int, default=100000, help="Hausdorff sampling count")
    parser.add_argument("--views", type=int, default=20, help="Number of render views for LFD/PSNR")
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for HD surface sampling")
    args = parser.parse_args()

    print(f"Loading meshes...")
    mesh_in = load_mesh(args.input_mesh)
    mesh_out = load_mesh(args.output_mesh)

    # --- Mesh info ---
    print("\n=== Mesh Info ===")
    mesh_info(mesh_in, "Input")
    mesh_info(mesh_out, "Output")

    # --- Watertight / Manifold ---
    print("\n=== Topology ===")
    in_stats = topology_diagnostics(mesh_in)
    out_stats = topology_diagnostics(mesh_out)
    in_wt = is_watertight(mesh_in)
    in_mf = in_stats["manifold"]
    out_wt = is_watertight(mesh_out)
    out_mf = out_stats["manifold"]
    print(f"  Input:  Watertight={in_wt}  Manifold={in_mf}")
    print(f"  Output: Watertight={out_wt}  Manifold={out_mf}")
    print_topology_diagnostics(in_stats, "Input")
    print_topology_diagnostics(out_stats, "Output")

    # --- Hausdorff Distance ---
    print("\n=== Metrics ===")
    print(f"  Computing Hausdorff distance ({args.samples} samples)...")
    hd = hausdorff_distance(mesh_in, mesh_out, args.samples, seed=args.seed)
    print(f"  HD:   {hd}")

    # --- Render views ---
    print(f"  Rendering {args.views} views at {args.resolution}x{args.resolution}...")
    cam_positions = _camera_positions(args.views)
    render_center, render_scale = _shared_render_frame(mesh_in, mesh_out)

    # LFD: both meshes rendered single-sided (front_only) so holes affect silhouette
    lfd_images_in = _render_views(
        mesh_in, cam_positions, render_center, render_scale,
        args.resolution, render_mode="front_only",
    )
    lfd_images_out = _render_views(
        mesh_out, cam_positions, render_center, render_scale,
        args.resolution, render_mode="front_only",
    )
    lfd = compute_lfd(lfd_images_in, lfd_images_out)
    print(f"  LFD:  {lfd}")

    # PSNR: input double-face (reference), output front white + back black
    psnr_images_in = _render_views(
        mesh_in, cam_positions, render_center, render_scale,
        args.resolution, render_mode="reference",
    )
    psnr_images_out = _render_views(
        mesh_out, cam_positions, render_center, render_scale,
        args.resolution, render_mode="backface_black",
    )
    psnr = compute_psnr(psnr_images_in, psnr_images_out)
    print(f"  PSNR: {psnr}")

    print("\nDone.")


if __name__ == "__main__":
    main()
