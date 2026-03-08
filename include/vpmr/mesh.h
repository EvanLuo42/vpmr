#pragma once

#include "vpmr/types.h"

namespace vpmr {

struct Mesh {
    MatXd V;
    MatXi F;

    VecXd visibility;
    VecXd orientation;
    VecXd openness;

    VecXi face_mapping;
    VecXi offset_source;

    MatXd TC;
    MatXi FTC;
    MatXd VC;

    double bbox_diagonal() const;
    double total_area() const;
    Vec3d face_normal(int f) const;
    double face_area(int f) const;
    Vec3d face_centroid(int f) const;
};

} // namespace vpmr
