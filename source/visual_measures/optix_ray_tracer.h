#pragma once

#include "vpmr/mesh.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <vector>

namespace vpmr {

struct RayResult {
    int face_id;
    float distance;
};

class OptixRayTracer {
public:
    OptixRayTracer();
    ~OptixRayTracer();

    void build_accel(const Mesh &mesh);

    std::vector<int> trace(const std::vector<Vec3d> &origins, const std::vector<Vec3d> &directions,
                                       int N_b, const Vec3d &bbox_min, const Vec3d &bbox_max
    ) const;

    // Upload per-face normals to GPU (call after build_accel)
    void upload_face_normals(const Mesh &mesh);

private:
    void init_optix();
    void create_module();
    void create_pipeline();
    void create_sbt();

    CUcontext cu_context_ = nullptr;
    OptixDeviceContext optix_context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;

    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup hitgroup_pg_ = nullptr;

    CUdeviceptr sbt_raygen_record_ = 0;
    CUdeviceptr sbt_miss_record_ = 0;
    CUdeviceptr sbt_hitgroup_record_ = 0;
    OptixShaderBindingTable sbt_ = {};

    OptixTraversableHandle gas_handle_ = 0;
    CUdeviceptr gas_buffer_ = 0;
    CUdeviceptr d_face_normals_ = 0;
};

} // namespace vpmr
