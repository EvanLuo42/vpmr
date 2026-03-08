#include "vpmr/mesh.h"
#include <igl/adjacency_list.h>
#include <igl/boundary_facets.h>
#include <igl/triangle_triangle_adjacency.h>
#include <map>
#include <vector>

namespace vpmr {

using Edge = std::pair<int, int>;

Edge make_edge(int a, int b) {
    return a < b ? Edge{a, b} : Edge{b, a};
}

std::map<Edge, std::vector<int>> build_edge_face_map(const MatXi &F) {
    std::map<Edge, std::vector<int>> ef;
    for (int f = 0; f < F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            Edge edge = make_edge(F(f, e), F(f, (e + 1) % 3));
            ef[edge].push_back(f);
        }
    }
    return ef;
}

double boundary_ratio(const Mesh &mesh) {
    auto ef = build_edge_face_map(mesh.F);
    int boundary_edges = 0;
    int total_edges = 0;
    for (auto &[e, faces] : ef) {
        ++total_edges;
        if (faces.size() == 1)
            ++boundary_edges;
    }
    return total_edges > 0 ? (double) boundary_edges / total_edges : 0.0;
}

} // namespace vpmr
