#include "topology.h"
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace vpmr {

using Edge = std::pair<int, int>;

static Edge make_edge(int a, int b) {
    return a < b ? Edge{a, b} : Edge{b, a};
}

static int opposite_vertex_on_edge(const MatXi &F, int face_id, int va, int vb) {
    int face[3] = {F(face_id, 0), F(face_id, 1), F(face_id, 2)};
    for (int i = 0; i < 3; ++i) {
        int u = face[i];
        int v = face[(i + 1) % 3];
        int w = face[(i + 2) % 3];
        if ((u == va && v == vb) || (u == vb && v == va))
            return w;
    }
    return -1;
}

static void remove_duplicate_faces_keep_one(Mesh &mesh) {
    int nf = mesh.F.rows();
    std::map<std::tuple<int, int, int>, int> first_face;
    std::vector<int> keep;
    keep.reserve(nf);

    for (int f = 0; f < nf; ++f) {
        int v[3] = {mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)};
        std::sort(v, v + 3);
        auto key = std::make_tuple(v[0], v[1], v[2]);
        if (first_face.find(key) == first_face.end()) {
            first_face[key] = f;
            keep.push_back(f);
        }
    }

    if ((int)keep.size() == nf)
        return;

    MatXi newF(keep.size(), 3);
    VecXi new_face_mapping;
    if (mesh.face_mapping.size() == nf)
        new_face_mapping.resize(keep.size());
    VecXi new_offset_source;
    if (mesh.offset_source.size() == nf)
        new_offset_source.resize(keep.size());

    for (int i = 0; i < (int)keep.size(); ++i) {
        int f = keep[i];
        newF.row(i) = mesh.F.row(f);
        if (new_face_mapping.size() > 0)
            new_face_mapping(i) = mesh.face_mapping(f);
        if (new_offset_source.size() > 0)
            new_offset_source(i) = mesh.offset_source(f);
    }

    mesh.F = newF;
    if (new_face_mapping.size() > 0)
        mesh.face_mapping = new_face_mapping;
    if (new_offset_source.size() > 0)
        mesh.offset_source = new_offset_source;

    std::cout << "  [topo] remove_duplicate_faces: dropped "
              << (nf - (int)keep.size()) << " duplicate faces" << std::endl;
}

static bool split_face_on_edge(
        const MatXi &F, int face_id, int va, int vb, int vm,
        Vec3i &tri0, Vec3i &tri1) {
    int face[3] = {F(face_id, 0), F(face_id, 1), F(face_id, 2)};
    for (int i = 0; i < 3; ++i) {
        int u = face[i];
        int v = face[(i + 1) % 3];
        int w = face[(i + 2) % 3];
        if ((u == va && v == vb) || (u == vb && v == va)) {
            tri0 = Vec3i(u, vm, w);
            tri1 = Vec3i(vm, v, w);
            return true;
        }
    }
    return false;
}

static void split_non_manifold_edges(Mesh &mesh) {
    // Split non-manifold edges by inserting a midpoint and splitting a pair of
    // incident triangles. This preserves even edge incidence instead of
    // detaching faces into open seams.
    int split_pairs = 0;
    bool saw_odd_incidence = false;
    int pass = 0;

    struct EdgeSplitOp {
        int v0, v1;
        int f0, f1;
    };

    while (true) {
        std::map<Edge, std::vector<int>> edge_faces;
        int nf = mesh.F.rows();
        for (int f = 0; f < nf; ++f) {
            for (int e = 0; e < 3; ++e) {
                edge_faces[make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3))].push_back(f);
            }
        }

        std::vector<char> face_locked(nf, false);
        std::vector<EdgeSplitOp> ops;
        for (auto &[edge, faces] : edge_faces) {
            if ((int) faces.size() <= 2)
                continue;

            if ((int) faces.size() % 2 != 0)
                saw_odd_incidence = true;

            int f0 = -1, f1 = -1;
            int opp0 = -1;
            for (int f : faces) {
                if (face_locked[f])
                    continue;
                if (f0 < 0) {
                    f0 = f;
                    opp0 = opposite_vertex_on_edge(mesh.F, f, edge.first, edge.second);
                } else {
                    int opp1 = opposite_vertex_on_edge(mesh.F, f, edge.first, edge.second);
                    if (opp1 != opp0) {
                        f1 = f;
                        break;
                    }
                }
            }

            if (f0 >= 0 && f1 < 0) {
                for (int f : faces) {
                    if (face_locked[f] || f == f0)
                        continue;
                    f1 = f;
                    break;
                }
            }

            if (f0 < 0 || f1 < 0)
                continue;

            face_locked[f0] = true;
            face_locked[f1] = true;
            ops.push_back({edge.first, edge.second, f0, f1});
        }

        if (ops.empty())
            break;

        ++pass;
        int old_nf = mesh.F.rows();
        int old_nv = mesh.V.rows();
        MatXi oldF = mesh.F;
        MatXd oldV = mesh.V;

        mesh.V.conservativeResize(old_nv + (int)ops.size(), 3);
        MatXi newF(old_nf + 2 * (int)ops.size(), 3);
        newF.topRows(old_nf) = oldF;

        VecXi new_face_mapping;
        if (mesh.face_mapping.size() == old_nf) {
            new_face_mapping.resize(old_nf + 2 * (int)ops.size());
            new_face_mapping.head(old_nf) = mesh.face_mapping;
        }
        VecXi new_offset_source;
        if (mesh.offset_source.size() == old_nf) {
            new_offset_source.resize(old_nf + 2 * (int)ops.size());
            new_offset_source.head(old_nf) = mesh.offset_source;
        }

        for (int i = 0; i < (int)ops.size(); ++i) {
            const auto &op = ops[i];
            int vm = old_nv + i;
            mesh.V.row(vm) = 0.5 * (oldV.row(op.v0) + oldV.row(op.v1));

            Vec3i f0a, f0b, f1a, f1b;
            bool ok0 = split_face_on_edge(oldF, op.f0, op.v0, op.v1, vm, f0a, f0b);
            bool ok1 = split_face_on_edge(oldF, op.f1, op.v0, op.v1, vm, f1a, f1b);
            if (!ok0 || !ok1) {
                newF.row(old_nf + 2 * i) = oldF.row(op.f0);
                newF.row(old_nf + 2 * i + 1) = oldF.row(op.f1);
                continue;
            }

            newF.row(op.f0) = f0a.transpose();
            newF.row(op.f1) = f1a.transpose();
            newF.row(old_nf + 2 * i) = f0b.transpose();
            newF.row(old_nf + 2 * i + 1) = f1b.transpose();

            if (new_face_mapping.size() > 0) {
                new_face_mapping(old_nf + 2 * i) = mesh.face_mapping(op.f0);
                new_face_mapping(old_nf + 2 * i + 1) = mesh.face_mapping(op.f1);
            }
            if (new_offset_source.size() > 0) {
                new_offset_source(old_nf + 2 * i) = mesh.offset_source(op.f0);
                new_offset_source(old_nf + 2 * i + 1) = mesh.offset_source(op.f1);
            }
        }

        mesh.F = newF;
        if (new_face_mapping.size() > 0)
            mesh.face_mapping = new_face_mapping;
        if (new_offset_source.size() > 0)
            mesh.offset_source = new_offset_source;

        split_pairs += (int)ops.size();
        std::cout << "  [topo] split_non_manifold_edges: pass " << pass
                  << ", split " << ops.size() << " face pairs" << std::endl;
    }

    if (split_pairs > 0) {
        std::cout << "  [topo] split_non_manifold_edges: total "
                  << split_pairs << " face pairs" << std::endl;
        if (saw_odd_incidence) {
            std::cout << "  [topo] split_non_manifold_edges: encountered "
                      << "odd-incidence edges; watertightness may still be impossible"
                      << std::endl;
        }
    }
}

static void split_non_manifold_vertices(Mesh &mesh) {
    int pass = 0;
    while (true) {
        int nf = mesh.F.rows();
        int nv = mesh.V.rows();

        // Build vertex → face adjacency
        std::vector<std::vector<int>> vf(nv);
        for (int f = 0; f < nf; ++f) {
            for (int j = 0; j < 3; ++j) {
                vf[mesh.F(f, j)].push_back(f);
            }
        }

        // Build face adjacency via shared edges
        std::map<Edge, std::vector<int>> edge_faces;
        for (int f = 0; f < nf; ++f) {
            for (int e = 0; e < 3; ++e) {
                edge_faces[make_edge(mesh.F(f, e), mesh.F(f, (e + 1) % 3))].push_back(f);
            }
        }

        struct VertexSplitOp {
            int old_v;
            int new_v;
            std::vector<int> faces;
        };

        std::vector<VertexSplitOp> ops;
        for (int v = 0; v < nv; ++v) {
            if (vf[v].size() <= 1)
                continue;

            // Check connectivity: flood-fill faces around v via shared edges containing v
            std::vector<std::vector<int>> components;

            std::set<int> remaining(vf[v].begin(), vf[v].end());
            while (!remaining.empty()) {
                std::vector<int> comp;
                std::queue<int> q;
                int start = *remaining.begin();
                q.push(start);
                remaining.erase(start);

                while (!q.empty()) {
                    int f = q.front();
                    q.pop();
                    comp.push_back(f);

                    for (int e = 0; e < 3; ++e) {
                        int a = mesh.F(f, e);
                        int b = mesh.F(f, (e + 1) % 3);
                        if (a != v && b != v)
                            continue;
                        Edge edge = make_edge(a, b);
                        for (int nb : edge_faces[edge]) {
                            if (remaining.count(nb)) {
                                remaining.erase(nb);
                                q.push(nb);
                            }
                        }
                    }
                }

                components.push_back(comp);
            }

            if (components.size() <= 1)
                continue;

            for (int c = 1; c < (int) components.size(); ++c) {
                ops.push_back({v, nv + (int)ops.size(), std::move(components[c])});
            }
        }

        if (ops.empty())
            break;

        ++pass;
        MatXi newF = mesh.F;
        mesh.V.conservativeResize(nv + (int)ops.size(), 3);
        for (const auto &op : ops) {
            mesh.V.row(op.new_v) = mesh.V.row(op.old_v);
            for (int f : op.faces) {
                for (int j = 0; j < 3; ++j) {
                    if (newF(f, j) == op.old_v)
                        newF(f, j) = op.new_v;
                }
            }
        }
        mesh.F = newF;
        std::cout << "  [topo] split_non_manifold_vertices: pass " << pass
                  << ", duplicated " << ops.size() << " vertices" << std::endl;
    }
}

static void remove_degenerate_faces(Mesh &mesh) {
    int nf = mesh.F.rows();
    std::vector<int> keep;
    keep.reserve(nf);
    for (int f = 0; f < nf; ++f) {
        if (mesh.F(f, 0) != mesh.F(f, 1) && mesh.F(f, 1) != mesh.F(f, 2) && mesh.F(f, 0) != mesh.F(f, 2))
            keep.push_back(f);
    }
    if ((int)keep.size() == nf)
        return;

    MatXi newF(keep.size(), 3);
    VecXi new_face_mapping;
    if (mesh.face_mapping.size() == nf)
        new_face_mapping.resize(keep.size());
    VecXi new_offset_source;
    if (mesh.offset_source.size() == nf)
        new_offset_source.resize(keep.size());

    for (int i = 0; i < (int)keep.size(); ++i) {
        int f = keep[i];
        newF.row(i) = mesh.F.row(f);
        if (new_face_mapping.size() > 0)
            new_face_mapping(i) = mesh.face_mapping(f);
        if (new_offset_source.size() > 0)
            new_offset_source(i) = mesh.offset_source(f);
    }

    std::cout << "  [topo] remove_degenerate_faces: dropped "
              << (nf - (int)keep.size()) << " degenerate faces" << std::endl;

    mesh.F = newF;
    if (new_face_mapping.size() > 0)
        mesh.face_mapping = new_face_mapping;
    if (new_offset_source.size() > 0)
        mesh.offset_source = new_offset_source;
}

void fix_topology(Mesh &mesh) {
    remove_degenerate_faces(mesh);
    remove_duplicate_faces_keep_one(mesh);
    split_non_manifold_edges(mesh);
    split_non_manifold_vertices(mesh);
}

} // namespace vpmr
