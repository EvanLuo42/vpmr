#include "simplification.h"
#include <igl/remove_unreferenced.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace vpmr {

using Edge = std::pair<int, int>;

static Edge make_edge(int a, int b) {
    return a < b ? Edge{a, b} : Edge{b, a};
}

static bool faces_coplanar(const MatXd &V, const MatXi &F, int f1, int f2, double tol = 1e-6) {
    Vec3d v0a(V.row(F(f1, 0))), v1a(V.row(F(f1, 1))), v2a(V.row(F(f1, 2)));
    Vec3d v0b(V.row(F(f2, 0))), v1b(V.row(F(f2, 1))), v2b(V.row(F(f2, 2)));
    Vec3d n1 = (v1a - v0a).cross(v2a - v0a);
    Vec3d n2 = (v1b - v0b).cross(v2b - v0b);
    double a1 = n1.norm(), a2 = n2.norm();
    if (a1 < 1e-15 || a2 < 1e-15)
        return false;
    n1 /= a1;
    n2 /= a2;
    if ((n1 - n2).norm() > tol)
        return false;
    // Same normal — check coplanar (same offset along normal)
    double d1 = n1.dot(v0a);
    double d2 = n1.dot(v0b);
    return std::abs(d1 - d2) < tol;
}

struct PatchInfo {
    std::vector<int> faces;
    Vec3d normal;
    int mapping;
};

static std::vector<PatchInfo> find_coplanar_patches(const Mesh &mesh) {
    int nf = mesh.F.rows();

    // Build edge -> face adjacency
    std::map<Edge, std::vector<int>> edge_faces;
    for (int f = 0; f < nf; ++f) {
        for (int e = 0; e < 3; ++e) {
            edge_faces[make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3))].push_back(f);
        }
    }

    // Flood-fill coplanar patches (§3.6.1: group adjacent co-planar faces
    // with consistent orientation, regardless of face_mapping)
    std::vector<int> patch_id(nf, -1);
    std::vector<PatchInfo> patches;

    for (int seed = 0; seed < nf; ++seed) {
        if (patch_id[seed] >= 0)
            continue;

        PatchInfo patch;
        patch.mapping = mesh.face_mapping.size() > 0 ? mesh.face_mapping(seed) : -1;

        int pid = (int) patches.size();
        patch_id[seed] = pid;
        patch.faces.push_back(seed);

        std::queue<int> q;
        q.push(seed);

        while (!q.empty()) {
            int f = q.front();
            q.pop();

            for (int e = 0; e < 3; ++e) {
                Edge edge = make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3));
                for (int nb : edge_faces[edge]) {
                    if (patch_id[nb] >= 0)
                        continue;

                    // Check coplanarity: sub-faces of any mapped original
                    // triangle are guaranteed coplanar. For extra faces or
                    // cross-mapping pairs, use numerical check.
                    int f_map = mesh.face_mapping.size() > 0 ? mesh.face_mapping(f) : -1;
                    int nb_map = mesh.face_mapping.size() > 0 ? mesh.face_mapping(nb) : -1;
                    bool coplanar;
                    if (f_map >= 0 && f_map == nb_map) {
                        coplanar = true; // same original face — guaranteed coplanar
                    } else {
                        coplanar = faces_coplanar(mesh.V, mesh.F, seed, nb);
                    }

                    if (coplanar) {
                        patch_id[nb] = pid;
                        patch.faces.push_back(nb);
                        q.push(nb);
                        // Track a valid face_mapping for the patch
                        if (patch.mapping < 0 && nb_map >= 0)
                            patch.mapping = nb_map;
                    }
                }
            }
        }

        // Compute area-weighted patch normal (robust even with tiny slivers)
        Vec3d weighted_normal = Vec3d::Zero();
        for (int f : patch.faces) {
            Vec3d v0 = mesh.V.row(mesh.F(f, 0));
            Vec3d v1 = mesh.V.row(mesh.F(f, 1));
            Vec3d v2 = mesh.V.row(mesh.F(f, 2));
            weighted_normal += (v1 - v0).cross(v2 - v0);
        }
        double wn_len = weighted_normal.norm();
        patch.normal = wn_len > 1e-15 ? weighted_normal / wn_len : mesh.face_normal(seed);

        patches.push_back(patch);
    }

    return patches;
}

// Find geometric boundary edges: edges shared by at least two different patches
static std::set<Edge> find_geometric_boundary_edges(
        const MatXi &F, const std::vector<int> &patch_id) {
    std::map<Edge, std::set<int>> edge_patches;
    int nf = F.rows();
    for (int f = 0; f < nf; ++f) {
        for (int e = 0; e < 3; ++e) {
            Edge edge = make_edge(F(f, e), F(f, (e + 1) % 3));
            edge_patches[edge].insert(patch_id[f]);
        }
    }
    std::set<Edge> boundary;
    for (auto &[edge, pids] : edge_patches) {
        if (pids.size() >= 2)
            boundary.insert(edge);
    }
    return boundary;
}

// Find UV boundary edges: edges where adjacent faces have different UV coordinates
static std::set<Edge> find_uv_boundary_edges(const Mesh &mesh) {
    std::set<Edge> uv_boundary;
    if (mesh.FTC.rows() == 0 || mesh.TC.rows() == 0)
        return uv_boundary;

    int nf = mesh.F.rows();
    // Build edge -> (face, local_edge_index) for UV lookup
    std::map<Edge, std::vector<std::pair<int, int>>> edge_face_local;
    for (int f = 0; f < nf; ++f) {
        for (int e = 0; e < 3; ++e) {
            Edge edge = make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3));
            edge_face_local[edge].push_back({f, e});
        }
    }

    double uv_tol = 1e-8;
    for (auto &[edge, fl_list] : edge_face_local) {
        if (fl_list.size() != 2)
            continue;
        auto [f1, e1] = fl_list[0];
        auto [f2, e2] = fl_list[1];

        // Get UV coords for this edge's two endpoints from each face
        int a = edge.first, b = edge.second;

        // Find which local vertices in f1 correspond to a,b
        using Vec2d = Eigen::Vector2d;
        auto get_uv = [&](int face, int vert) -> Vec2d {
            for (int i = 0; i < 3; ++i) {
                if (mesh.F(face, i) == vert) {
                    int tc_idx = mesh.FTC(face, i);
                    return Vec2d(mesh.TC(tc_idx, 0), mesh.TC(tc_idx, 1));
                }
            }
            return Vec2d(0, 0);
        };

        Vec2d uv_a1 = get_uv(f1, a), uv_a2 = get_uv(f2, a);
        Vec2d uv_b1 = get_uv(f1, b), uv_b2 = get_uv(f2, b);

        if ((uv_a1 - uv_a2).norm() > uv_tol || (uv_b1 - uv_b2).norm() > uv_tol) {
            uv_boundary.insert(edge);
        }
    }
    return uv_boundary;
}

// Collect constraint edges for a patch: geometric + UV boundary edges that touch patch vertices
static std::set<Edge> patch_constraint_edges(
        const std::vector<int> &patch_faces, const MatXi &F,
        const std::set<Edge> &geo_boundary, const std::set<Edge> &uv_boundary) {
    std::set<int> patch_verts;
    for (int f : patch_faces)
        for (int i = 0; i < 3; ++i)
            patch_verts.insert(F(f, i));

    std::set<Edge> constraints;
    auto add_if_in_patch = [&](const std::set<Edge> &src) {
        for (auto &e : src) {
            if (patch_verts.count(e.first) && patch_verts.count(e.second))
                constraints.insert(e);
        }
    };
    add_if_in_patch(geo_boundary);
    add_if_in_patch(uv_boundary);
    return constraints;
}

struct BoundaryResult {
    std::vector<int> outer_loop;
    std::vector<std::vector<int>> inner_loops;
};

static BoundaryResult extract_boundary_loops(const std::vector<int> &faces, const MatXi &F) {
    BoundaryResult result;

    // Find boundary edges of the patch (count == 1 within patch)
    std::map<Edge, int> edge_count;
    std::map<std::pair<int, int>, bool> directed_edges;

    for (int f : faces) {
        for (int e = 0; e < 3; ++e) {
            int a = F(f, e);
            int b = F(f, (e + 1) % 3);
            edge_count[make_edge(a, b)]++;
            directed_edges[{a, b}] = true;
        }
    }

    // Build directed adjacency for boundary
    std::map<int, std::vector<int>> adj;
    for (auto &[edge, count] : edge_count) {
        if (count == 1) {
            if (directed_edges.count({edge.first, edge.second})) {
                adj[edge.first].push_back(edge.second);
            }
            if (directed_edges.count({edge.second, edge.first})) {
                adj[edge.second].push_back(edge.first);
            }
        }
    }

    if (adj.empty())
        return result;

    // Extract all loops
    std::set<int> globally_visited;
    std::vector<std::vector<int>> all_loops;

    for (auto &[start_v, _] : adj) {
        if (globally_visited.count(start_v))
            continue;

        std::vector<int> loop;
        int current = start_v;
        std::set<int> visited;

        do {
            loop.push_back(current);
            visited.insert(current);
            globally_visited.insert(current);
            bool found = false;
            for (int next : adj[current]) {
                if (visited.find(next) == visited.end()) {
                    current = next;
                    found = true;
                    break;
                }
            }
            if (!found)
                break;
        } while (current != start_v && (int) loop.size() < (int) adj.size() + 1);

        if (loop.size() >= 3)
            all_loops.push_back(loop);
    }

    if (all_loops.empty())
        return result;

    // The outer loop has the largest absolute signed area in the projection plane
    // For simplicity, pick the loop with the most vertices as outer
    int outer_idx = 0;
    for (int i = 1; i < (int) all_loops.size(); ++i) {
        if (all_loops[i].size() > all_loops[outer_idx].size())
            outer_idx = i;
    }
    result.outer_loop = all_loops[outer_idx];
    for (int i = 0; i < (int) all_loops.size(); ++i) {
        if (i != outer_idx)
            result.inner_loops.push_back(all_loops[i]);
    }

    return result;
}

// Collapse degree-2 vertices on collinear edges in constraint set (§3.6.2)
static void collapse_collinear_boundary_vertices(
        std::vector<int> &loop, const MatXd &V,
        const std::set<Edge> &constraints, double tol = 1e-6) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < (int) loop.size(); ++i) {
            int n = (int) loop.size();
            if (n <= 3)
                break;
            int prev = (i - 1 + n) % n;
            int next = (i + 1) % n;
            int vp = loop[prev], vi = loop[i], vn = loop[next];

            // Check that both adjacent edges are constraint edges
            Edge e1 = make_edge(vp, vi), e2 = make_edge(vi, vn);
            if (!constraints.count(e1) || !constraints.count(e2))
                continue;

            // Check collinearity
            Vec3d d1 = (V.row(vi) - V.row(vp)).normalized();
            Vec3d d2 = (V.row(vn) - V.row(vi)).normalized();
            if (d1.cross(d2).norm() < tol) {
                loop.erase(loop.begin() + i);
                changed = true;
                break;
            }
        }
    }
}

// Check if a triangle intersects another triangle (Möller–Trumbore style)
static bool triangles_intersect(
        const Vec3d &a0, const Vec3d &a1, const Vec3d &a2,
        const Vec3d &b0, const Vec3d &b1, const Vec3d &b2) {
    // Use separating axis theorem with triangle normals and edge cross products
    auto tri_normal = [](const Vec3d &v0, const Vec3d &v1, const Vec3d &v2) {
        return (v1 - v0).cross(v2 - v0);
    };

    auto project_tri = [](const Vec3d &axis, const Vec3d &v0, const Vec3d &v1, const Vec3d &v2,
                          double &lo, double &hi) {
        double d0 = axis.dot(v0), d1 = axis.dot(v1), d2 = axis.dot(v2);
        lo = std::min({d0, d1, d2});
        hi = std::max({d0, d1, d2});
    };

    auto separated = [&](const Vec3d &axis) -> bool {
        if (axis.squaredNorm() < 1e-30)
            return false;
        double lo1, hi1, lo2, hi2;
        project_tri(axis, a0, a1, a2, lo1, hi1);
        project_tri(axis, b0, b1, b2, lo2, hi2);
        return hi1 < lo2 - 1e-10 || hi2 < lo1 - 1e-10;
    };

    Vec3d na = tri_normal(a0, a1, a2);
    Vec3d nb = tri_normal(b0, b1, b2);

    if (separated(na) || separated(nb))
        return false;

    Vec3d edges_a[3] = {a1 - a0, a2 - a1, a0 - a2};
    Vec3d edges_b[3] = {b1 - b0, b2 - b1, b0 - b2};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Vec3d axis = edges_a[i].cross(edges_b[j]);
            if (separated(axis))
                return false;
        }
    }

    return true;
}

static bool is_ear(const std::vector<int> &loop, int i, const MatXd &V, const Vec3d &normal,
                   const std::set<Edge> &constraints) {
    int n = (int) loop.size();
    int prev = (i - 1 + n) % n;
    int next = (i + 1) % n;

    int vp = loop[prev], vi = loop[i], vn = loop[next];

    // Cannot clip an ear if the edge prev->next is a constraint (would remove a boundary)
    if (constraints.count(make_edge(vp, vn)))
        return false;

    Vec3d v0 = V.row(vp);
    Vec3d v1 = V.row(vi);
    Vec3d v2 = V.row(vn);

    Vec3d cross = (v1 - v0).cross(v2 - v0);
    if (cross.dot(normal) <= 0)
        return false;

    // Check no other loop vertices inside triangle
    for (int j = 0; j < n; ++j) {
        if (j == prev || j == i || j == next)
            continue;
        Vec3d p = V.row(loop[j]);
        Vec3d e0 = v1 - v0, e1 = v2 - v0, e2 = p - v0;
        double d00 = e0.dot(e0), d01 = e0.dot(e1), d11 = e1.dot(e1);
        double d20 = e2.dot(e0), d21 = e2.dot(e1);
        double denom = d00 * d11 - d01 * d01;
        if (std::abs(denom) < 1e-15)
            continue;
        double u = (d11 * d20 - d01 * d21) / denom;
        double v = (d00 * d21 - d01 * d20) / denom;
        if (u >= -1e-10 && v >= -1e-10 && u + v <= 1.0 + 1e-10)
            return false;
    }

    return true;
}

// Self-intersection check: does new_tri intersect any face in check_faces within extended bbox?
static bool ear_causes_intersection(
        const Vec3i &new_tri, const MatXd &V, const MatXi &F,
        const std::vector<int> &check_faces, double l_extended) {
    Vec3d t0 = V.row(new_tri(0)), t1 = V.row(new_tri(1)), t2 = V.row(new_tri(2));

    // Extended bounding box of the new triangle
    Vec3d lo = t0.cwiseMin(t1).cwiseMin(t2).array() - l_extended;
    Vec3d hi = t0.cwiseMax(t1).cwiseMax(t2).array() + l_extended;

    std::set<int> tri_verts = {new_tri(0), new_tri(1), new_tri(2)};

    for (int f : check_faces) {
        // Skip faces sharing a vertex (adjacent faces can't meaningfully intersect here)
        bool shares_vert = false;
        for (int i = 0; i < 3; ++i) {
            if (tri_verts.count(F(f, i))) {
                shares_vert = true;
                break;
            }
        }
        if (shares_vert)
            continue;

        Vec3d f0 = V.row(F(f, 0)), f1 = V.row(F(f, 1)), f2 = V.row(F(f, 2));

        // Quick bbox check
        Vec3d flo = f0.cwiseMin(f1).cwiseMin(f2);
        Vec3d fhi = f0.cwiseMax(f1).cwiseMax(f2);
        if ((flo.array() > hi.array()).any() || (fhi.array() < lo.array()).any())
            continue;

        if (triangles_intersect(t0, t1, t2, f0, f1, f2))
            return true;
    }
    return false;
}

static std::vector<Vec3i> ear_clip(
        std::vector<int> loop, const MatXd &V, const Vec3d &normal,
        const std::set<Edge> &constraints,
        const MatXi &F, const std::vector<int> &nearby_faces, double l_extended) {
    std::vector<Vec3i> tris;

    while (loop.size() > 3) {
        bool clipped = false;
        for (int i = 0; i < (int) loop.size(); ++i) {
            if (is_ear(loop, i, V, normal, constraints)) {
                int n = (int) loop.size();
                int prev = (i - 1 + n) % n;
                int next = (i + 1) % n;
                Vec3i tri(loop[prev], loop[i], loop[next]);

                // Self-intersection sanity check (§3.6.2)
                if (l_extended > 0 && ear_causes_intersection(tri, V, F, nearby_faces, l_extended)) {
                    continue;
                }

                tris.push_back(tri);
                loop.erase(loop.begin() + i);
                clipped = true;
                break;
            }
        }
        if (!clipped)
            break;
    }

    if (loop.size() == 3) {
        tris.push_back(Vec3i(loop[0], loop[1], loop[2]));
    }

    return tris;
}

// Collect faces near a patch for self-intersection checks (within extended bbox)
static std::vector<int> collect_nearby_faces(
        const std::vector<int> &patch_faces, const Mesh &mesh, double l_extended) {
    // Compute patch bounding box
    std::set<int> patch_verts;
    std::set<int> patch_face_set(patch_faces.begin(), patch_faces.end());
    for (int f : patch_faces)
        for (int i = 0; i < 3; ++i)
            patch_verts.insert(mesh.F(f, i));

    Vec3d lo(1e30, 1e30, 1e30), hi(-1e30, -1e30, -1e30);
    for (int v : patch_verts) {
        lo = lo.cwiseMin(Vec3d(mesh.V.row(v)));
        hi = hi.cwiseMax(Vec3d(mesh.V.row(v)));
    }
    lo.array() -= l_extended;
    hi.array() += l_extended;

    std::vector<int> nearby;
    int nf = mesh.F.rows();
    for (int f = 0; f < nf; ++f) {
        if (patch_face_set.count(f))
            continue;
        Vec3d f0 = mesh.V.row(mesh.F(f, 0));
        Vec3d f1 = mesh.V.row(mesh.F(f, 1));
        Vec3d f2 = mesh.V.row(mesh.F(f, 2));
        Vec3d flo = f0.cwiseMin(f1).cwiseMin(f2);
        Vec3d fhi = f0.cwiseMax(f1).cwiseMax(f2);
        if ((flo.array() > hi.array()).any() || (fhi.array() < lo.array()).any())
            continue;
        nearby.push_back(f);
    }
    return nearby;
}

void simplify_mesh(Mesh &mesh, const Config &config) {
    auto patches = find_coplanar_patches(mesh);

    // Build global patch_id per face
    int nf = mesh.F.rows();
    std::vector<int> patch_id(nf, -1);
    for (int p = 0; p < (int) patches.size(); ++p)
        for (int f : patches[p].faces)
            patch_id[f] = p;

    // Detect boundary edges (§3.6.1)
    auto geo_boundary = find_geometric_boundary_edges(mesh.F, patch_id);
    auto uv_boundary = find_uv_boundary_edges(mesh);

    double D = mesh.bbox_diagonal();
    double l_extended = D / config.l_extended_divisor;

    std::vector<Vec3i> new_faces;
    std::vector<int> new_face_mapping;
    std::vector<int> new_offset_source;

    for (auto &patch : patches) {
        if (patch.faces.size() <= 1) {
            for (int f : patch.faces) {
                new_faces.push_back(Vec3i(mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)));
                new_face_mapping.push_back(mesh.face_mapping.size() > 0 ? mesh.face_mapping(f) : -1);
                new_offset_source.push_back(mesh.offset_source.size() > 0 ? mesh.offset_source(f) : -1);
            }
            continue;
        }

        auto boundary = extract_boundary_loops(patch.faces, mesh.F);
        if (boundary.outer_loop.size() < 3) {
            for (int f : patch.faces) {
                new_faces.push_back(Vec3i(mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)));
                new_face_mapping.push_back(mesh.face_mapping.size() > 0 ? mesh.face_mapping(f) : -1);
                new_offset_source.push_back(mesh.offset_source.size() > 0 ? mesh.offset_source(f) : -1);
            }
            continue;
        }

        // The current ear clipping path only handles a single simple boundary.
        // Simplifying holed patches here would drop the inner loops and open cracks,
        // so keep those patches unchanged until hole-aware triangulation is added.
        if (!boundary.inner_loops.empty()) {
            for (int f : patch.faces) {
                new_faces.push_back(Vec3i(mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)));
                new_face_mapping.push_back(mesh.face_mapping.size() > 0 ? mesh.face_mapping(f) : -1);
                new_offset_source.push_back(mesh.offset_source.size() > 0 ? mesh.offset_source(f) : -1);
            }
            continue;
        }

        // Constraint edges for this patch
        auto constraints = patch_constraint_edges(patch.faces, mesh.F, geo_boundary, uv_boundary);

        // Edge collapse pre-pass (§3.6.2): remove degree-2 vertices on collinear constraint edges
        auto loop = boundary.outer_loop;
        collapse_collinear_boundary_vertices(loop, mesh.V, constraints, 1e-6);

        // Collect nearby faces for self-intersection check
        auto nearby = collect_nearby_faces(patch.faces, mesh, l_extended);

        auto tris = ear_clip(loop, mesh.V, patch.normal, constraints, mesh.F, nearby, l_extended);

        // Also triangulate inner loops (holes) — add their faces back as-is
        // Inner loops represent holes in the patch that we can't simplify away
        bool use_simplified = !tris.empty() && (int) tris.size() <= (int) patch.faces.size();

        if (!use_simplified) {
            for (int f : patch.faces) {
                new_faces.push_back(Vec3i(mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)));
                new_face_mapping.push_back(mesh.face_mapping.size() > 0 ? mesh.face_mapping(f) : -1);
                new_offset_source.push_back(mesh.offset_source.size() > 0 ? mesh.offset_source(f) : -1);
            }
        } else {
            for (auto &tri : tris) {
                new_faces.push_back(tri);

                // Determine face_mapping for this new triangle by finding
                // which original patch face contains its centroid
                Vec3d centroid = (mesh.V.row(tri(0)) + mesh.V.row(tri(1)) + mesh.V.row(tri(2))) / 3.0;
                int best_mapping = -1;
                int best_offset = -1;
                double best_dist = 1e30;
                for (int pf : patch.faces) {
                    int fm = mesh.face_mapping.size() > 0 ? mesh.face_mapping(pf) : -1;
                    if (fm < 0) continue;
                    Vec3d fc = mesh.face_centroid(pf);
                    double d = (centroid - fc).squaredNorm();
                    if (d < best_dist) {
                        best_dist = d;
                        best_mapping = fm;
                        best_offset = mesh.offset_source.size() > 0 ? mesh.offset_source(pf) : -1;
                    }
                }
                if (best_mapping < 0) {
                    best_mapping = patch.mapping;
                    best_offset = patch.faces.size() > 0 && mesh.offset_source.size() > 0
                            ? mesh.offset_source(patch.faces[0]) : -1;
                }
                new_face_mapping.push_back(best_mapping);
                new_offset_source.push_back(best_offset);
            }
        }
    }

    mesh.F.resize(new_faces.size(), 3);
    mesh.face_mapping.resize(new_faces.size());
    mesh.offset_source.resize(new_faces.size());

    for (int i = 0; i < (int) new_faces.size(); ++i) {
        mesh.F.row(i) = new_faces[i].transpose();
        mesh.face_mapping(i) = new_face_mapping[i];
        mesh.offset_source(i) = new_offset_source[i];
    }

    // Remove unreferenced vertices
    Eigen::MatrixXd V_clean;
    Eigen::MatrixXi F_clean;
    Eigen::VectorXi I_map;
    igl::remove_unreferenced(MatXd(mesh.V), MatXi(mesh.F), V_clean, F_clean, I_map);
    mesh.V = V_clean;
    mesh.F = F_clean;
}

} // namespace vpmr
