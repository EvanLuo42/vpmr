#pragma once

#include <cuda_runtime.h>

namespace vpmr {

struct LaunchParams {
    OptixTraversableHandle traversable;
    float3 *ray_origins;
    float3 *ray_directions;
    int *hit_face_ids;
    float *hit_distances;
    int num_rays;

    int N_b;
    float3 bbox_min;
    float3 bbox_max;
    float3 *face_normals;
    int *escaped;
};

} // namespace vpmr
