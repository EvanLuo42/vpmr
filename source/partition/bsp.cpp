#include "bsp.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/number_utils.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <queue>
#include <set>

namespace vpmr {

// ============================================================
// Helpers
// ============================================================

int BSPTree::add_vertex(const EPoint3 &p) {
    int idx = (int)vertices.size();
    vertices.push_back(p);
    vertices_d.push_back({CGAL::to_double(p.x()),
                          CGAL::to_double(p.y()),
                          CGAL::to_double(p.z())});
    return idx;
}

const EPlane3 &BSPTree::get_face_plane(BSPFace &f) {
    if (!f.has_plane) {
        f.cached_plane = EPlane3(vertices[f.verts[0]],
                                 vertices[f.verts[1]],
                                 vertices[f.verts[2]]);
        f.has_plane = true;
    }
    return f.cached_plane;
}

DAABB BSPTree::compute_cell_bbox(int cell_id) {
    BSPCell &cell = cells[cell_id];
    if (cell.bbox_valid)
        return cell.bbox;
    DAABB box;
    for (auto &f : cell.faces) {
        for (int v : f.verts) {
            box.expand(vertices_d[v][0], vertices_d[v][1], vertices_d[v][2]);
        }
    }
    cell.bbox = box;
    cell.bbox_valid = true;
    return box;
}

// ============================================================
// Delaunay initialization
// ============================================================

using Vb = CGAL::Triangulation_vertex_base_with_info_3<int, EK>;
using Tds = CGAL::Triangulation_data_structure_3<Vb>;
using DT = CGAL::Delaunay_triangulation_3<EK, Tds>;

void BSPTree::init_delaunay(const Eigen::MatrixXd &V) {
    int nv = V.rows();
    vertices.reserve(nv * 2);
    vertices_d.reserve(nv * 2);
    for (int i = 0; i < nv; ++i) {
        vertices.push_back(EPoint3(V(i, 0), V(i, 1), V(i, 2)));
        vertices_d.push_back({V(i, 0), V(i, 1), V(i, 2)});
    }

    // Build Delaunay triangulation with vertex info = original index
    std::vector<std::pair<EPoint3, int>> pts(nv);
    for (int i = 0; i < nv; ++i)
        pts[i] = {vertices[i], i};

    DT dt(pts.begin(), pts.end());

    // Map DT cell handles to BSP cell indices
    std::map<DT::Cell_handle, int> cell_map;
    int ncells = 0;
    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
        cell_map[cit] = ncells++;
    }
    cells.resize(ncells);

    // Build cells with faces and adjacency
    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
        int ci = cell_map[cit];
        BSPCell &cell = cells[ci];
        cell.faces.resize(4);

        for (int i = 0; i < 4; ++i) {
            BSPFace &face = cell.faces[i];
            face.label = -1;

            // Face opposite vertex i: vertices j,k,l (j<k<l from {0,1,2,3}\{i})
            int idx[3], k = 0;
            for (int j = 0; j < 4; ++j) {
                if (j != i)
                    idx[k++] = j;
            }

            int vi0 = cit->vertex(idx[0])->info();
            int vi1 = cit->vertex(idx[1])->info();
            int vi2 = cit->vertex(idx[2])->info();

            // Reversed CGAL order: face normal points from this cell toward neighbor
            face.verts = {vi0, vi2, vi1};

            DT::Cell_handle nb = cit->neighbor(i);
            if (dt.is_infinite(nb)) {
                face.neighbor = -1;
            } else {
                face.neighbor = cell_map[nb];
            }
        }
    }

    std::cout << "[BSP] Initialized with " << ncells << " Delaunay tets, "
              << nv << " vertices" << std::endl;
}

// ============================================================
// Edge-plane intersection (LPI)
// ============================================================

int BSPTree::intersect_edge_plane(int va, int vb, const EPlane3 &plane) {
    auto edge_key = std::make_pair(std::min(va, vb), std::max(va, vb));
    auto cache_key = std::make_pair(edge_key, current_face_id);
    auto it = edge_plane_cache.find(cache_key);
    if (it != edge_plane_cache.end())
        return it->second;

    const EPoint3 &A = vertices[va];
    const EPoint3 &B = vertices[vb];

    auto dA = plane.a() * A.x() + plane.b() * A.y() + plane.c() * A.z() + plane.d();
    auto dB = plane.a() * B.x() + plane.b() * B.y() + plane.c() * B.z() + plane.d();

    auto denom = dA - dB;
    auto x = (B.x() * dA - A.x() * dB) / denom;
    auto y = (B.y() * dA - A.y() * dB) / denom;
    auto z = (B.z() * dA - A.z() * dB) / denom;

    EPoint3 p(x, y, z);
    int new_v = add_vertex(p);
    edge_plane_cache[cache_key] = new_v;
    return new_v;
}

// ============================================================
// Order polygon vertices CCW on a plane
// ============================================================

void BSPTree::order_polygon_ccw(std::vector<int> &poly, const EPlane3 &plane) {
    if (poly.size() <= 3)
        return;

    auto n = plane.orthogonal_vector();

    EK::Vector_3 ref(1, 0, 0);
    if (CGAL::abs(n.x()) > CGAL::abs(n.y()) && CGAL::abs(n.x()) > CGAL::abs(n.z()))
        ref = EK::Vector_3(0, 1, 0);

    auto b1 = CGAL::cross_product(n, ref);
    auto b2 = CGAL::cross_product(n, b1);

    EK::FT cx(0), cy(0), cz(0);
    for (int v : poly) {
        cx += vertices[v].x();
        cy += vertices[v].y();
        cz += vertices[v].z();
    }
    int np = (int)poly.size();
    EPoint3 center(cx / np, cy / np, cz / np);

    struct ProjVert {
        int idx;
        EK::FT u, v;
    };
    std::vector<ProjVert> pv(np);
    for (int i = 0; i < np; ++i) {
        auto d = vertices[poly[i]] - center;
        pv[i].idx = poly[i];
        pv[i].u = d * b1;
        pv[i].v = d * b2;
    }

    std::sort(pv.begin(), pv.end(), [](const ProjVert &a, const ProjVert &b) {
        auto quad = [](const ProjVert &p) -> int {
            if (p.u > 0 && p.v >= 0) return 0;
            if (p.u <= 0 && p.v > 0) return 1;
            if (p.u < 0 && p.v <= 0) return 2;
            return 3;
        };
        int qa = quad(a), qb = quad(b);
        if (qa != qb)
            return qa < qb;
        return a.u * b.v - a.v * b.u > 0;
    });

    for (int i = 0; i < np; ++i)
        poly[i] = pv[i].idx;
}

// ============================================================
// Split cell along plane
// ============================================================

std::pair<int, int> BSPTree::split_cell(int cell_id, const EPlane3 &plane, int face_label) {
    BSPCell &cell = cells[cell_id];
    const int query_id = current_face_id + 1;

    // Collect unique vertices and classify
    struct LocalSign {
        int vertex;
        CGAL::Sign sign;
    };
    std::vector<LocalSign> signs;
    signs.reserve(32);

    auto get_local_sign = [&](int v) -> CGAL::Sign {
        for (const auto &entry : signs) {
            if (entry.vertex == v)
                return entry.sign;
        }

        CGAL::Sign s;
        if (v < (int)query_vertex_stamp.size() && query_vertex_stamp[v] == query_id) {
            s = query_vertex_sign[v];
        } else {
            s = plane.oriented_side(vertices[v]);
        }
        signs.push_back({v, s});
        return s;
    };

    auto set_local_sign = [&](int v, CGAL::Sign s) {
        for (auto &entry : signs) {
            if (entry.vertex == v) {
                entry.sign = s;
                return;
            }
        }
        signs.push_back({v, s});
    };

    auto lookup_local_sign = [&](int v, CGAL::Sign fallback) -> CGAL::Sign {
        for (const auto &entry : signs) {
            if (entry.vertex == v)
                return entry.sign;
        }
        return fallback;
    };

    for (auto &f : cell.faces) {
        for (int v : f.verts) {
            (void)get_local_sign(v);
        }
    }

    bool has_pos = false, has_neg = false;
    for (const auto &entry : signs) {
        if (entry.sign == CGAL::POSITIVE)
            has_pos = true;
        if (entry.sign == CGAL::NEGATIVE)
            has_neg = true;
    }
    if (!has_pos || !has_neg)
        return {-1, -1};

    // Label co-planar faces: if an existing face (label == -1) has all vertices
    // on the cut plane, it's co-planar with the input face being inserted.
    if (face_label >= 0) {
        for (auto &f : cell.faces) {
            if (f.label >= 0)
                continue;
            bool all_zero = true;
            for (int v : f.verts) {
                if (lookup_local_sign(v, CGAL::ZERO) != CGAL::ZERO) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero)
                f.label = face_label;
        }
    }

    // Compute edge-plane intersections
    struct LocalEdgeIsect {
        int v0;
        int v1;
        int isect;
    };
    std::vector<LocalEdgeIsect> edge_isect;
    edge_isect.reserve(32);

    auto find_edge_isect = [&](int a, int b) -> int {
        const int lo = std::min(a, b);
        const int hi = std::max(a, b);
        for (const auto &entry : edge_isect) {
            if (entry.v0 == lo && entry.v1 == hi)
                return entry.isect;
        }
        return -1;
    };

    for (auto &f : cell.faces) {
        int n = (int)f.verts.size();
        for (int i = 0; i < n; ++i) {
            int u = f.verts[i], v = f.verts[(i + 1) % n];
            CGAL::Sign su = lookup_local_sign(u, CGAL::ZERO);
            CGAL::Sign sv = lookup_local_sign(v, CGAL::ZERO);
            if (su != CGAL::ZERO && sv != CGAL::ZERO && su != sv) {
                if (find_edge_isect(u, v) < 0) {
                    int nv = intersect_edge_plane(u, v, plane);
                    edge_isect.push_back({std::min(u, v), std::max(u, v), nv});
                    set_local_sign(nv, CGAL::ZERO);
                    if (nv >= (int)query_vertex_stamp.size()) {
                        query_vertex_stamp.resize(vertices.size(), 0);
                        query_vertex_sign.resize(vertices.size(), CGAL::ZERO);
                    }
                    query_vertex_stamp[nv] = query_id;
                    query_vertex_sign[nv] = CGAL::ZERO;
                }
            }
        }
    }

    // Split each face into positive and negative parts
    BSPCell pos_cell, neg_cell;
    std::vector<int> cut_verts;

    for (auto &f : cell.faces) {
        std::vector<int> pos_v, neg_v;
        int n = (int)f.verts.size();

        for (int i = 0; i < n; ++i) {
            int u = f.verts[i];
            int v = f.verts[(i + 1) % n];
            CGAL::Sign su = lookup_local_sign(u, CGAL::ZERO);

            if (su == CGAL::POSITIVE) {
                pos_v.push_back(u);
            } else if (su == CGAL::NEGATIVE) {
                neg_v.push_back(u);
            } else {
                pos_v.push_back(u);
                neg_v.push_back(u);
                cut_verts.push_back(u);
            }

            int isect = find_edge_isect(u, v);
            if (isect >= 0) {
                pos_v.push_back(isect);
                neg_v.push_back(isect);
                cut_verts.push_back(isect);
            }
        }

        if ((int)pos_v.size() >= 3) {
            BSPFace pf;
            pf.verts = pos_v;
            pf.label = f.label;
            pf.neighbor = f.neighbor;
            // Inherit cached plane if the face wasn't split (same vertices)
            if ((int)pos_v.size() == n && f.has_plane) {
                pf.cached_plane = f.cached_plane;
                pf.has_plane = true;
            }
            pos_cell.faces.push_back(pf);
        }
        if ((int)neg_v.size() >= 3) {
            BSPFace nf;
            nf.verts = neg_v;
            nf.label = f.label;
            nf.neighbor = f.neighbor;
            if ((int)neg_v.size() == n && f.has_plane) {
                nf.cached_plane = f.cached_plane;
                nf.has_plane = true;
            }
            neg_cell.faces.push_back(nf);
        }
    }

    // Deduplicate cut_verts
    std::sort(cut_verts.begin(), cut_verts.end());
    cut_verts.erase(std::unique(cut_verts.begin(), cut_verts.end()), cut_verts.end());

    if ((int)cut_verts.size() < 3) {
        return {-1, -1};
    }

    order_polygon_ccw(cut_verts, plane);

    int neg_id = (int)cells.size();

    // Add cut face to positive cell (normal must point outward → toward neg_id → negative side)
    {
        BSPFace cf;
        cf.verts = std::vector<int>(cut_verts.rbegin(), cut_verts.rend());
        cf.label = face_label;
        cf.neighbor = neg_id;
        cf.cached_plane = plane.opposite();
        cf.has_plane = true;
        pos_cell.faces.push_back(cf);
    }

    // Add cut face to negative cell (normal must point outward → toward cell_id → positive side)
    {
        BSPFace cf;
        cf.verts = cut_verts;
        cf.label = face_label;
        cf.neighbor = cell_id;
        cf.cached_plane = plane;
        cf.has_plane = true;
        neg_cell.faces.push_back(cf);
    }

    // Invalidate bboxes (will be recomputed on demand)
    pos_cell.bbox_valid = false;
    neg_cell.bbox_valid = false;

    // Store cells
    cells[cell_id] = pos_cell;
    cells.push_back(neg_cell);

    // Update neighbors
    for (auto &f : cells[neg_id].faces) {
        if (f.neighbor < 0 || f.neighbor == cell_id)
            continue;

        BSPCell &nb = cells[f.neighbor];
        std::set<int> f_vset(f.verts.begin(), f.verts.end());

        for (auto &nf : nb.faces) {
            if (nf.neighbor != cell_id)
                continue;
            int shared = 0;
            for (int v : nf.verts) {
                if (f_vset.count(v))
                    ++shared;
            }
            if (shared >= 2) {
                bool all_neg_or_zero = true;
                for (int v : nf.verts) {
                    if (lookup_local_sign(v, CGAL::ZERO) == CGAL::POSITIVE) {
                        all_neg_or_zero = false;
                        break;
                    }
                }
                if (all_neg_or_zero) {
                    nf.neighbor = neg_id;
                }
            }
        }
        nb.bbox_valid = false; // neighbor might reference new vertices
    }

    // Split neighbor faces that straddle the cut plane
    for (auto &pf : cells[cell_id].faces) {
        if (pf.neighbor < 0 || pf.neighbor == neg_id)
            continue;
        int nb_id = pf.neighbor;
        BSPCell &nb = cells[nb_id];

        for (int fi = 0; fi < (int)nb.faces.size(); ++fi) {
            BSPFace &nf = nb.faces[fi];
            if (nf.neighbor != cell_id)
                continue;

            bool has_neg_v = false;
            for (int v : nf.verts) {
                if (lookup_local_sign(v, CGAL::ZERO) == CGAL::NEGATIVE) {
                    has_neg_v = true;
                    break;
                }
            }

            if (has_neg_v) {
                std::vector<int> nb_pos_v, nb_neg_v;
                int nn = (int)nf.verts.size();
                for (int i = 0; i < nn; ++i) {
                    int u = nf.verts[i];
                    int v = nf.verts[(i + 1) % nn];
                    auto su = lookup_local_sign(u, CGAL::ZERO);

                    if (su == CGAL::POSITIVE) {
                        nb_pos_v.push_back(u);
                    } else if (su == CGAL::NEGATIVE) {
                        nb_neg_v.push_back(u);
                    } else {
                        nb_pos_v.push_back(u);
                        nb_neg_v.push_back(u);
                    }

                    int isect = find_edge_isect(u, v);
                    if (isect >= 0) {
                        nb_pos_v.push_back(isect);
                        nb_neg_v.push_back(isect);
                    }
                }

                if ((int)nb_pos_v.size() >= 3 && (int)nb_neg_v.size() >= 3) {
                    nf.verts = nb_pos_v;
                    nf.neighbor = cell_id;
                    nf.has_plane = false; // invalidate

                    BSPFace neg_face;
                    neg_face.verts = nb_neg_v;
                    neg_face.label = nf.label;
                    neg_face.neighbor = neg_id;
                    nb.faces.push_back(neg_face);
                }
                nb.bbox_valid = false;
                break;
            }
        }
    }

    return {cell_id, neg_id};
}

// ============================================================
// Walk from hint cell to cell containing point (Stochastic Walk)
// Uses cached planes for speed.
// ============================================================

int BSPTree::walk_to_point(const EPoint3 &p, int hint) const {
    if (hint < 0 || hint >= (int)cells.size() || cells[hint].faces.empty())
        hint = 0;

    int current = hint;
    int max_steps = (int)cells.size();
    for (int step = 0; step < max_steps; ++step) {
        const BSPCell &cell = cells[current];
        if (cell.faces.empty())
            break;

        bool inside = true;
        int best_neighbor = -1;
        for (auto &f : cell.faces) {
            if (f.verts.size() < 3)
                continue;
            // Use cached plane if available, otherwise construct
            EPlane3 fp = f.has_plane ? f.cached_plane
                                     : EPlane3(vertices[f.verts[0]],
                                               vertices[f.verts[1]],
                                               vertices[f.verts[2]]);
            if (fp.oriented_side(p) == CGAL::POSITIVE) {
                inside = false;
                if (f.neighbor >= 0) {
                    best_neighbor = f.neighbor;
                    break;
                }
            }
        }
        if (inside)
            return current;
        if (best_neighbor < 0)
            break;
        current = best_neighbor;
    }
    return -1;
}

// ============================================================
// Find cells intersecting a triangle
// Uses AABB culling to avoid expanding BFS along the infinite plane.
// ============================================================

std::vector<int> BSPTree::find_intersecting_cells(int v0, int v1, int v2) {
    std::vector<int> result;
    if (cells.empty())
        return result;

    const int query_id = current_face_id + 1;
    if ((int)query_vertex_stamp.size() < (int)vertices.size()) {
        query_vertex_stamp.resize(vertices.size(), 0);
        query_vertex_sign.resize(vertices.size(), CGAL::ZERO);
    }
    if ((int)query_cell_stamp.size() < (int)cells.size()) {
        query_cell_stamp.resize(cells.size(), 0);
    }

    // Compute triangle AABB (double precision, fast)
    DAABB tri_box;
    tri_box.expand(vertices_d[v0][0], vertices_d[v0][1], vertices_d[v0][2]);
    tri_box.expand(vertices_d[v1][0], vertices_d[v1][1], vertices_d[v1][2]);
    tri_box.expand(vertices_d[v2][0], vertices_d[v2][1], vertices_d[v2][2]);
    // Small epsilon expansion for numerical safety
    for (int i = 0; i < 3; ++i) {
        tri_box.lo[i] -= 1e-10;
        tri_box.hi[i] += 1e-10;
    }

    EPlane3 tri_plane(vertices[v0], vertices[v1], vertices[v2]);

    auto get_sign = [&](int v) -> CGAL::Sign {
        if (query_vertex_stamp[v] != query_id) {
            query_vertex_stamp[v] = query_id;
            query_vertex_sign[v] = tri_plane.oriented_side(vertices[v]);
        }
        return query_vertex_sign[v];
    };

    auto classify_cell = [&](int cell_id, bool &touches_plane) -> bool {
        bool has_pos = false;
        bool has_neg = false;
        bool has_zero = false;

        for (auto &f : cells[cell_id].faces) {
            for (int v : f.verts) {
                CGAL::Sign s = get_sign(v);
                if (s == CGAL::POSITIVE) has_pos = true;
                else if (s == CGAL::NEGATIVE) has_neg = true;
                else has_zero = true;
            }
        }

        const bool straddles_plane = has_pos && has_neg;
        touches_plane = straddles_plane || has_zero;
        return straddles_plane;
    };

    // Find starting cell by walking from hint
    EPoint3 centroid(
        (vertices[v0].x() + vertices[v1].x() + vertices[v2].x()) / 3,
        (vertices[v0].y() + vertices[v1].y() + vertices[v2].y()) / 3,
        (vertices[v0].z() + vertices[v1].z() + vertices[v2].z()) / 3);

    int start = walk_to_point(centroid, last_split_cell);

    if (start < 0) {
        // Fallback: check all cells with AABB + plane test
        for (int i = 0; i < (int)cells.size(); ++i) {
            if (cells[i].faces.empty())
                continue;
            DAABB cbox = compute_cell_bbox(i);
            if (!cbox.overlaps(tri_box))
                continue;

            bool touches_plane = false;
            if (classify_cell(i, touches_plane))
                result.push_back(i);
        }
        return result;
    }

    // BFS from starting cell, with AABB culling
    std::queue<int> queue;
    queue.push(start);
    query_cell_stamp[start] = query_id;

    while (!queue.empty()) {
        int ci = queue.front();
        queue.pop();

        // AABB cull: skip cells whose bbox doesn't overlap the triangle's bbox
        DAABB cbox = compute_cell_bbox(ci);
        if (!cbox.overlaps(tri_box))
            continue;

        bool touches_plane = false;
        if (classify_cell(ci, touches_plane)) {
            result.push_back(ci);
        }

        // Only expand cells that still touch the triangle's supporting plane.
        if (touches_plane) {
            for (auto &f : cells[ci].faces) {
                if (f.neighbor >= 0 &&
                    f.neighbor < (int)query_cell_stamp.size() &&
                    query_cell_stamp[f.neighbor] != query_id) {
                    query_cell_stamp[f.neighbor] = query_id;
                    queue.push(f.neighbor);
                }
            }
        }
    }

    return result;
}

// ============================================================
// Insert face
// ============================================================

void BSPTree::insert_face(int face_id, int v0, int v1, int v2) {
    current_face_id = face_id;
    edge_plane_cache.clear();

    EPlane3 plane(vertices[v0], vertices[v1], vertices[v2]);

    auto intersecting = find_intersecting_cells(v0, v1, v2);

    for (int ci : intersecting) {
        auto [pos_id, neg_id] = split_cell(ci, plane, face_id);
        if (pos_id >= 0)
            last_split_cell = pos_id;
    }
}

// ============================================================
// Extract partition
// ============================================================

void BSPTree::extract_partition(
    Eigen::MatrixXd &V_out,
    Eigen::MatrixXi &F_out,
    Eigen::MatrixXi &per_face_cells,
    Eigen::VectorXi &face_labels,
    int &num_cells,
    std::vector<int> *new_to_old_cells) {

    // Convert exact vertices to double
    int nv = (int)vertices.size();
    V_out.resize(nv, 3);
    for (int i = 0; i < nv; ++i) {
        V_out(i, 0) = vertices_d[i][0];
        V_out(i, 1) = vertices_d[i][1];
        V_out(i, 2) = vertices_d[i][2];
    }

    struct OutputFace {
        int v0, v1, v2;
        int pos_cell, neg_cell;
        int label;
    };
    std::vector<OutputFace> out_faces;

    for (int ci = 0; ci < (int)cells.size(); ++ci) {
        for (auto &f : cells[ci].faces) {
            int nb = f.neighbor;

            // Dedup shared faces: only emit from the cell with the smaller index.
            // Boundary faces (nb < 0) are not shared, so always emit them.
            if (nb >= 0 && ci > nb)
                continue;

            for (int i = 1; i + 1 < (int)f.verts.size(); ++i) {
                OutputFace of;
                of.v0 = f.verts[0];
                of.v1 = f.verts[i];
                of.v2 = f.verts[i + 1];
                of.pos_cell = (nb >= 0) ? nb : -1;
                of.neg_cell = ci;
                of.label = f.label;
                out_faces.push_back(of);
            }
        }
    }

    // Remap cell indices
    std::set<int> used_cells;
    for (auto &of : out_faces) {
        if (of.pos_cell >= 0) used_cells.insert(of.pos_cell);
        if (of.neg_cell >= 0) used_cells.insert(of.neg_cell);
    }

    std::map<int, int> cell_remap;
    cell_remap[-1] = 0;
    int next_id = 1;
    for (int c : used_cells) {
        cell_remap[c] = next_id++;
    }
    num_cells = next_id;
    if (new_to_old_cells != nullptr) {
        new_to_old_cells->assign(num_cells, -1);
        for (int c : used_cells) {
            (*new_to_old_cells)[cell_remap[c]] = c;
        }
    }

    int nf = (int)out_faces.size();
    F_out.resize(nf, 3);
    per_face_cells.resize(nf, 2);
    face_labels.resize(nf);

    for (int i = 0; i < nf; ++i) {
        F_out(i, 0) = out_faces[i].v0;
        F_out(i, 1) = out_faces[i].v1;
        F_out(i, 2) = out_faces[i].v2;
        per_face_cells(i, 0) = cell_remap[out_faces[i].pos_cell];
        per_face_cells(i, 1) = cell_remap[out_faces[i].neg_cell];
        face_labels(i) = out_faces[i].label;
    }
}

Eigen::Vector3d BSPTree::cell_representative(int cell_id) const {
    if (cell_id < 0 || cell_id >= (int)cells.size() || cells[cell_id].faces.empty())
        return Eigen::Vector3d::Zero();

    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;
    std::set<int> unique_verts;
    for (const auto &f : cells[cell_id].faces) {
        for (int v : f.verts) {
            if (unique_verts.insert(v).second) {
                sum.x() += vertices_d[v][0];
                sum.y() += vertices_d[v][1];
                sum.z() += vertices_d[v][2];
                ++count;
            }
        }
    }
    if (count == 0)
        return Eigen::Vector3d::Zero();
    return sum / (double)count;
}

} // namespace vpmr
