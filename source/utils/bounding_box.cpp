#include "bounding_box.h"

namespace vpmr {

AABB AABB::from_mesh_vertices(const MatXd &V, double padding) {
    AABB box;
    box.min_pt = V.colwise().minCoeff().transpose() - Vec3d::Constant(padding);
    box.max_pt = V.colwise().maxCoeff().transpose() + Vec3d::Constant(padding);
    return box;
}

bool AABB::contains(const Vec3d &p) const {
    return (p.array() >= min_pt.array()).all() && (p.array() <= max_pt.array()).all();
}

} // namespace vpmr
