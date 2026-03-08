#include "offset.h"
#include <igl/boundary_loop.h>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace vpmr {

using Edge = std::pair<int, int>;

static Edge make_edge(int a, int b) {
    return a < b ? Edge{a, b} : Edge{b, a};
}

static double boundary_ratio(const Mesh &mesh) {
    std::map<Edge, int> edge_count;
    for (int f = 0; f < mesh.F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            edge_count[make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3))]++;
        }
    }
    int boundary = 0;
    for (auto &[e, c] : edge_count) {
        if (c == 1)
            ++boundary;
    }
    return edge_count.empty() ? 0.0 : (double) boundary / edge_count.size();
}

static void hole_fill(Mesh &mesh) {
    std::vector<std::vector<int>> loops;
    igl::boundary_loop(mesh.F, loops);

    int orig_nf = mesh.F.rows();
    int added = 0;
    for (auto &loop : loops) {
        added += (int) loop.size() - 2;
    }

    MatXi newF(orig_nf + added, 3);
    newF.topRows(orig_nf) = mesh.F;

    VecXi new_offset_source(orig_nf + added);
    if (mesh.offset_source.size() > 0)
        new_offset_source.head(orig_nf) = mesh.offset_source;
    else
        new_offset_source.head(orig_nf).setConstant(-1);

    int fi = orig_nf;
    for (auto &loop : loops) {
        for (int i = 1; i + 1 < (int) loop.size(); ++i) {
            newF.row(fi) << loop[0], loop[i + 1], loop[i];
            new_offset_source(fi) = -1;
            ++fi;
        }
    }

    mesh.F = newF;
    mesh.offset_source = new_offset_source;
}

void offset_open_surfaces(Mesh &mesh, const Config &config) {
    int nf = mesh.F.rows();
    int nv = mesh.V.rows();

    double br = boundary_ratio(mesh);
    if (br < config.boundary_ratio_threshold) {
        hole_fill(mesh);
        return;
    }

    // Classify open faces
    std::vector<bool> is_open(nf, false);
    for (int f = 0; f < nf; ++f) {
        if (mesh.openness.size() > f && mesh.openness(f) > config.epsilon_openness) {
            is_open[f] = true;
        }
    }

    // Find connected patches of open faces with consistent orientation
    std::map<Edge, std::vector<int>> edge_faces;
    for (int f = 0; f < nf; ++f) {
        if (!is_open[f])
            continue;
        for (int e = 0; e < 3; ++e) {
            edge_faces[make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3))].push_back(f);
        }
    }

    std::vector<int> patch_id(nf, -1);
    int num_patches = 0;
    for (int f = 0; f < nf; ++f) {
        if (!is_open[f] || patch_id[f] >= 0)
            continue;
        int pid = num_patches++;
        std::queue<int> q;
        q.push(f);
        patch_id[f] = pid;
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            for (int e = 0; e < 3; ++e) {
                int u = mesh.F(cur, e), v = mesh.F(cur, (e + 1) % 3);
                Edge edge = make_edge(u, v);
                for (int nb : edge_faces[edge]) {
                    if (patch_id[nb] >= 0)
                        continue;
                    // Only group faces with consistent orientation:
                    // cur has directed edge (u,v), nb must have (v,u)
                    bool consistent = false;
                    for (int ne = 0; ne < 3; ++ne) {
                        if (mesh.F(nb, ne) == v && mesh.F(nb, (ne + 1) % 3) == u) {
                            consistent = true;
                            break;
                        }
                    }
                    if (consistent) {
                        patch_id[nb] = pid;
                        q.push(nb);
                    }
                }
            }
        }
    }

    if (num_patches == 0)
        return;

    double d_offset = mesh.bbox_diagonal() / config.offset_divisor;

    // Compute per-vertex offset direction (area-weighted avg of adjacent open face normals)
    std::vector<Vec3d> vert_offset_dir(nv, Vec3d::Zero());

    for (int f = 0; f < nf; ++f) {
        if (!is_open[f])
            continue;
        Vec3d n = mesh.face_normal(f);
        double a = mesh.face_area(f);
        for (int i = 0; i < 3; ++i) {
            int v = mesh.F(f, i);
            vert_offset_dir[v] += n * a;
        }
    }

    std::set<int> open_verts;
    for (int f = 0; f < nf; ++f) {
        if (!is_open[f])
            continue;
        for (int i = 0; i < 3; ++i)
            open_verts.insert(mesh.F(f, i));
    }

    // Create offset vertices
    std::map<int, int> vert_map; // original vertex → offset vertex
    int new_nv = nv;
    MatXd newV(nv + (int) open_verts.size(), 3);
    newV.topRows(nv) = mesh.V;

    for (int v : open_verts) {
        Vec3d dir = vert_offset_dir[v];
        if (dir.norm() > 1e-10)
            dir.normalize();
        newV.row(new_nv) = mesh.V.row(v) - (dir * d_offset).transpose();
        vert_map[v] = new_nv;
        ++new_nv;
    }
    newV.conservativeResize(new_nv, 3);
    mesh.V = newV;

    // Find boundary edges of open patches for side-stitching
    struct BoundaryEdge {
        int u, v; // directed edge as it appears in the original face
    };
    std::vector<BoundaryEdge> boundary_edges;

    for (int f = 0; f < nf; ++f) {
        if (!is_open[f])
            continue;
        for (int e = 0; e < 3; ++e) {
            int u = mesh.F(f, e), v = mesh.F(f, (e + 1) % 3);
            Edge ue = make_edge(u, v);
            // Count faces in the same patch sharing this edge
            int same_patch = 0;
            for (int nb : edge_faces[ue]) {
                if (patch_id[nb] == patch_id[f])
                    ++same_patch;
            }
            if (same_patch == 1) {
                boundary_edges.push_back({u, v});
            }
        }
    }

    // Create offset faces (reversed winding) + side-stitching faces
    int open_count = 0;
    for (int f = 0; f < nf; ++f) {
        if (is_open[f])
            ++open_count;
    }

    int side_count = (int) boundary_edges.size() * 2; // 2 triangles per quad
    int total_f = nf + open_count + side_count;
    MatXi newF(total_f, 3);
    newF.topRows(nf) = mesh.F;

    VecXi new_offset_source(total_f);
    if (mesh.offset_source.size() > 0)
        new_offset_source.head(nf) = mesh.offset_source;
    else
        new_offset_source.head(nf).setConstant(-1);

    int fi = nf;
    for (int f = 0; f < nf; ++f) {
        if (!is_open[f])
            continue;
        // Reversed winding for offset face
        newF(fi, 0) = vert_map[mesh.F(f, 0)];
        newF(fi, 1) = vert_map[mesh.F(f, 2)];
        newF(fi, 2) = vert_map[mesh.F(f, 1)];
        new_offset_source(fi) = f;
        ++fi;
    }

    // Side-stitching: connect original boundary edge to offset boundary edge
    // For directed edge (u,v), the face is to its left; outside is to its right.
    // Side quad winding (CCW from outside): v, u, map[u], map[v]
    for (auto &be : boundary_edges) {
        int u = be.u, v = be.v;
        int mu = vert_map[u], mv = vert_map[v];
        newF(fi, 0) = v;
        newF(fi, 1) = u;
        newF(fi, 2) = mu;
        new_offset_source(fi) = -1;
        ++fi;
        newF(fi, 0) = v;
        newF(fi, 1) = mu;
        newF(fi, 2) = mv;
        new_offset_source(fi) = -1;
        ++fi;
    }

    mesh.F = newF;
    mesh.offset_source = new_offset_source;
}

} // namespace vpmr
