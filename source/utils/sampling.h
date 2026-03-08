#pragma once

#include "vpmr/mesh.h"
#include <vector>

namespace vpmr {

struct SamplePoint {
    int face_id;
    Vec3d position;
    Vec3d normal;
};

std::vector<SamplePoint> sample_faces(const Mesh &mesh, int N_min, int N_total);

} // namespace vpmr
