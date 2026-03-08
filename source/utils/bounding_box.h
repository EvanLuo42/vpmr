#pragma once

#include "vpmr/types.h"

namespace vpmr {

struct AABB {
    Vec3d min_pt;
    Vec3d max_pt;

    static AABB from_mesh_vertices(const MatXd &V, double padding = 0.0);
    bool contains(const Vec3d &p) const;
};

} // namespace vpmr
