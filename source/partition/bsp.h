#pragma once

#include "vpmr/types.h"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <array>
#include <map>
#include <vector>

namespace vpmr {

using EK = CGAL::Exact_predicates_exact_constructions_kernel;
using EPoint3 = EK::Point_3;
using EPlane3 = EK::Plane_3;

// Double-precision AABB for fast culling
struct DAABB {
    double lo[3] = {1e30, 1e30, 1e30};
    double hi[3] = {-1e30, -1e30, -1e30};

    void expand(double x, double y, double z) {
        if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
        if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
        if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
    }
    bool overlaps(const DAABB &o) const {
        return lo[0] <= o.hi[0] && hi[0] >= o.lo[0] &&
               lo[1] <= o.hi[1] && hi[1] >= o.lo[1] &&
               lo[2] <= o.hi[2] && hi[2] >= o.lo[2];
    }
};

struct BSPFace {
    std::vector<int> verts; // vertex indices, CCW from positive side
    int label;              // potential input face index, or -1
    int neighbor;           // adjacent cell index, or -1 for convex hull boundary
    EPlane3 cached_plane;   // supporting plane (cached to avoid reconstruction)
    bool has_plane = false;
};

struct BSPCell {
    std::vector<BSPFace> faces;
    DAABB bbox;             // cached bounding box (double precision)
    bool bbox_valid = false;
};

class BSPTree {
public:
    std::vector<EPoint3> vertices;
    // Double-precision mirror of vertices for fast AABB computation
    std::vector<std::array<double, 3>> vertices_d;
    std::vector<BSPCell> cells;

    // Initialize from Delaunay tetrahedralization of given vertices
    void init_delaunay(const Eigen::MatrixXd &V);

    // Insert a face into the BSP tree
    void insert_face(int face_id, int v0, int v1, int v2);

    // Extract the final partition as triangulated faces with cell adjacency
    void extract_partition(
        Eigen::MatrixXd &V_out,
        Eigen::MatrixXi &F_out,
        Eigen::MatrixXi &per_face_cells,
        Eigen::VectorXi &face_labels,
        int &num_cells,
        std::vector<int> *new_to_old_cells = nullptr);

    // A representative point inside the convex cell, approximated by averaging its vertices.
    Eigen::Vector3d cell_representative(int cell_id) const;

private:
    // Split cell along plane. Returns (pos_cell_id, neg_cell_id) or (-1,-1) if no split.
    std::pair<int, int> split_cell(int cell_id, const EPlane3 &plane, int face_label);

    // Find all cells that triangle (v0,v1,v2) intersects
    std::vector<int> find_intersecting_cells(int v0, int v1, int v2);

    // Compute edge-plane intersection, returns new vertex index
    int intersect_edge_plane(int va, int vb, const EPlane3 &plane);

    // Order vertices of a convex polygon on a plane in CCW order (from positive side)
    void order_polygon_ccw(std::vector<int> &poly, const EPlane3 &plane);

    // Walk from a hint cell to find cell containing a point
    int walk_to_point(const EPoint3 &p, int hint) const;

    // Get or compute the AABB for a cell (caches result in BSPCell::bbox)
    DAABB compute_cell_bbox(int cell_id);

    // Get or compute the cached plane for a face
    const EPlane3 &get_face_plane(BSPFace &f);

    // Add a new vertex (exact + double mirror)
    int add_vertex(const EPoint3 &p);

    // Cache for edge-plane intersections (to avoid duplicates)
    std::map<std::pair<std::pair<int,int>, int>, int> edge_plane_cache;
    int current_face_id = -1; // for cache keying by insertion round
    int last_split_cell = 0;  // hint for point location
    std::vector<int> query_vertex_stamp;
    std::vector<CGAL::Sign> query_vertex_sign;
    std::vector<int> query_cell_stamp;
};

} // namespace vpmr
