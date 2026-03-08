#include "optix_ray_tracer.h"
#include "launch_params.h"
#include "embedded_ptx.h"
#include <cuda.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace vpmr
{

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

static void check_cuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

static void check_optix(OptixResult res, const char* msg)
{
    if (res != OPTIX_SUCCESS)
    {
        throw std::runtime_error(std::string(msg) + ": " + optixGetErrorString(res));
    }
}

OptixRayTracer::OptixRayTracer()
{
    init_optix();
    create_module();
    create_pipeline();
    create_sbt();
}

OptixRayTracer::~OptixRayTracer()
{
    if (d_face_normals_)
        cudaFree((void*)d_face_normals_);
    if (sbt_raygen_record_)
        cudaFree((void*)sbt_raygen_record_);
    if (sbt_miss_record_)
        cudaFree((void*)sbt_miss_record_);
    if (sbt_hitgroup_record_)
        cudaFree((void*)sbt_hitgroup_record_);
    if (gas_buffer_)
        cudaFree((void*)gas_buffer_);
    if (pipeline_)
        optixPipelineDestroy(pipeline_);
    if (raygen_pg_)
        optixProgramGroupDestroy(raygen_pg_);
    if (miss_pg_)
        optixProgramGroupDestroy(miss_pg_);
    if (hitgroup_pg_)
        optixProgramGroupDestroy(hitgroup_pg_);
    if (module_)
        optixModuleDestroy(module_);
    if (optix_context_)
        optixDeviceContextDestroy(optix_context_);
}

void OptixRayTracer::init_optix()
{
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "CUDA init");

    CUresult cu_res = cuCtxGetCurrent(&cu_context_);
    if (cu_res != CUDA_SUCCESS)
        throw std::runtime_error("cuCtxGetCurrent failed");

    check_optix(optixInit(), "optixInit");
    check_optix(optixDeviceContextCreate(cu_context_, nullptr, &optix_context_), "optixDeviceContextCreate");
}

void OptixRayTracer::create_module()
{
    OptixModuleCompileOptions module_opts = {};
    module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_opts = {};
    pipeline_opts.usesMotionBlur = false;
    pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_opts.numPayloadValues = 2;
    pipeline_opts.numAttributeValues = 2;
    pipeline_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_opts.pipelineLaunchParamsVariableName = "params";

    const char* ptx = optix_programs_ptx;
    size_t ptx_size = strlen(ptx);

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(
        optixModuleCreate(optix_context_, &module_opts, &pipeline_opts, ptx, ptx_size, log, &log_size, &module_), "optixModuleCreate"
    );
}

void OptixRayTracer::create_pipeline()
{
    OptixProgramGroupOptions pg_opts = {};
    char log[2048];
    size_t log_size;

    // Raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module_;
    raygen_desc.raygen.entryFunctionName = "__raygen__trace";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(optix_context_, &raygen_desc, 1, &pg_opts, log, &log_size, &raygen_pg_), "raygen PG");

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__trace";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(optix_context_, &miss_desc, 1, &pg_opts, log, &log_size, &miss_pg_), "miss PG");

    // Hitgroup
    OptixProgramGroupDesc hg_desc = {};
    hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hg_desc.hitgroup.moduleCH = module_;
    hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__trace";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(optix_context_, &hg_desc, 1, &pg_opts, log, &log_size, &hitgroup_pg_), "hitgroup PG");

    // Pipeline
    OptixProgramGroup pgs[] = {raygen_pg_, miss_pg_, hitgroup_pg_};
    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 1;

    OptixPipelineCompileOptions pipeline_opts = {};
    pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_opts.numPayloadValues = 2;
    pipeline_opts.numAttributeValues = 2;
    pipeline_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_opts.pipelineLaunchParamsVariableName = "params";

    log_size = sizeof(log);
    check_optix(optixPipelineCreate(optix_context_, &pipeline_opts, &link_opts, pgs, 3, log, &log_size, &pipeline_), "pipeline");
}

void OptixRayTracer::create_sbt()
{
    // Raygen record
    SbtRecord raygen_record;
    optixSbtRecordPackHeader(raygen_pg_, &raygen_record);
    check_cuda(cudaMalloc((void**)&sbt_raygen_record_, sizeof(SbtRecord)), "sbt raygen malloc");
    check_cuda(cudaMemcpy((void*)sbt_raygen_record_, &raygen_record, sizeof(SbtRecord), cudaMemcpyHostToDevice), "sbt raygen copy");

    // Miss record
    SbtRecord miss_record;
    optixSbtRecordPackHeader(miss_pg_, &miss_record);
    check_cuda(cudaMalloc((void**)&sbt_miss_record_, sizeof(SbtRecord)), "sbt miss malloc");
    check_cuda(cudaMemcpy((void*)sbt_miss_record_, &miss_record, sizeof(SbtRecord), cudaMemcpyHostToDevice), "sbt miss copy");

    // Hitgroup record
    SbtRecord hg_record;
    optixSbtRecordPackHeader(hitgroup_pg_, &hg_record);
    check_cuda(cudaMalloc((void**)&sbt_hitgroup_record_, sizeof(SbtRecord)), "sbt hg malloc");
    check_cuda(cudaMemcpy((void*)sbt_hitgroup_record_, &hg_record, sizeof(SbtRecord), cudaMemcpyHostToDevice), "sbt hg copy");

    sbt_.raygenRecord = sbt_raygen_record_;
    sbt_.missRecordBase = sbt_miss_record_;
    sbt_.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = sbt_hitgroup_record_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.hitgroupRecordCount = 1;
}

void OptixRayTracer::build_accel(const Mesh& mesh)
{
    if (gas_buffer_)
    {
        cudaFree((void*)gas_buffer_);
        gas_buffer_ = 0;
    }

    int nf = static_cast<int>(mesh.F.rows());
    std::vector<float3> vertices(mesh.V.rows());
    for (int i = 0; i < mesh.V.rows(); ++i)
    {
        vertices[i] = make_float3(static_cast<float>(mesh.V(i, 0)), static_cast<float>(mesh.V(i, 1)), static_cast<float>(mesh.V(i, 2)));
    }

    std::vector<uint3> indices(nf);
    for (int i = 0; i < nf; ++i)
    {
        indices[i] = make_uint3(mesh.F(i, 0), mesh.F(i, 1), mesh.F(i, 2));
    }

    CUdeviceptr d_vertices, d_indices;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices.size() * sizeof(float3)), "vertex malloc");
    check_cuda(
        cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice),
        "vertex copy"
    );
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices.size() * sizeof(uint3)), "index malloc");
    check_cuda(
        cudaMemcpy(reinterpret_cast<void*>(d_indices), indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice), "index copy"
    );

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.numVertices = static_cast<uint>(vertices.size());
    build_input.triangleArray.vertexBuffers = &d_vertices;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.numIndexTriplets = static_cast<uint>(nf);
    build_input.triangleArray.indexBuffer = d_indices;

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.triangleArray.flags = &flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buf_sizes;
    check_optix(optixAccelComputeMemoryUsage(optix_context_, &accel_opts, &build_input, 1, &buf_sizes), "mem usage");

    CUdeviceptr d_temp;
    check_cuda(cudaMalloc((void**)&d_temp, buf_sizes.tempSizeInBytes), "temp malloc");
    check_cuda(cudaMalloc((void**)&gas_buffer_, buf_sizes.outputSizeInBytes), "gas malloc");

    check_optix(
        optixAccelBuild(
            optix_context_,
            nullptr,
            &accel_opts,
            &build_input,
            1,
            d_temp,
            buf_sizes.tempSizeInBytes,
            gas_buffer_,
            buf_sizes.outputSizeInBytes,
            &gas_handle_,
            nullptr,
            0
        ),
        "accel build"
    );

    cudaFree(reinterpret_cast<void*>(d_temp));
    cudaFree(reinterpret_cast<void*>(d_vertices));
    cudaFree(reinterpret_cast<void*>(d_indices));
}

void OptixRayTracer::upload_face_normals(const Mesh& mesh)
{
    if (d_face_normals_)
    {
        cudaFree(reinterpret_cast<void*>(d_face_normals_));
        d_face_normals_ = 0;
    }
    const int nf = static_cast<int>(mesh.F.rows());
    std::vector<float3> normals(nf);
    for (int i = 0; i < nf; ++i)
    {
        Vec3d n = mesh.face_normal(i);
        normals[i] = make_float3(static_cast<float>(n.x()), static_cast<float>(n.y()), static_cast<float>(n.z()));
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_face_normals_), nf * sizeof(float3)), "normals malloc");
    check_cuda(
        cudaMemcpy(reinterpret_cast<void*>(d_face_normals_), normals.data(), nf * sizeof(float3), cudaMemcpyHostToDevice), "normals copy"
    );
}

std::vector<int> OptixRayTracer::trace(
    const std::vector<Vec3d>& origins,
    const std::vector<Vec3d>& directions,
    const int N_b,
    const Vec3d& bbox_min,
    const Vec3d& bbox_max
) const
{
    const int n = static_cast<int>(origins.size());
    std::vector<float3> h_origins(n), h_dirs(n);
    for (int i = 0; i < n; ++i)
    {
        h_origins[i] =
            make_float3(static_cast<float>(origins[i].x()), static_cast<float>(origins[i].y()), static_cast<float>(origins[i].z()));
        h_dirs[i] = make_float3(
            static_cast<float>(directions[i].x()), static_cast<float>(directions[i].y()), static_cast<float>(directions[i].z())
        );
    }

    constexpr int BATCH_SIZE = 1 << 24; // 16M rays per launch
    const int alloc_size = std::min(n, BATCH_SIZE);

    CUdeviceptr d_origins, d_dirs, d_escaped, d_params;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_origins), alloc_size * sizeof(float3)), "ray origin malloc");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_dirs), alloc_size * sizeof(float3)), "ray dir malloc");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_escaped), alloc_size * sizeof(int)), "escaped malloc");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)), "params malloc");

    std::vector<int> h_escaped(n);

    for (int offset = 0; offset < n; offset += BATCH_SIZE)
    {
        int batch = std::min(BATCH_SIZE, n - offset);

        check_cuda(
            cudaMemcpy(reinterpret_cast<void*>(d_origins), h_origins.data() + offset, batch * sizeof(float3), cudaMemcpyHostToDevice), "ray origin copy"
        );
        check_cuda(
            cudaMemcpy(reinterpret_cast<void*>(d_dirs), h_dirs.data() + offset, batch * sizeof(float3), cudaMemcpyHostToDevice), "ray dir copy"
        );

        LaunchParams lp = {};
        lp.traversable = gas_handle_;
        lp.ray_origins = reinterpret_cast<float3*>(d_origins);
        lp.ray_directions = reinterpret_cast<float3*>(d_dirs);
        lp.hit_face_ids = nullptr;
        lp.hit_distances = nullptr;
        lp.num_rays = batch;
        lp.N_b = N_b;
        lp.bbox_min = make_float3(static_cast<float>(bbox_min.x()), static_cast<float>(bbox_min.y()), static_cast<float>(bbox_min.z()));
        lp.bbox_max = make_float3(static_cast<float>(bbox_max.x()), static_cast<float>(bbox_max.y()), static_cast<float>(bbox_max.z()));
        lp.face_normals = reinterpret_cast<float3*>(d_face_normals_);
        lp.escaped = reinterpret_cast<int*>(d_escaped);

        check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_params), &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice), "params copy");
        check_optix(optixLaunch(pipeline_, nullptr, d_params, sizeof(LaunchParams), &sbt_, batch, 1, 1), "launch");
        check_cuda(cudaDeviceSynchronize(), "sync");

        check_cuda(
            cudaMemcpy(h_escaped.data() + offset, reinterpret_cast<void*>(d_escaped), batch * sizeof(int), cudaMemcpyDeviceToHost), "escaped readback"
        );
    }

    cudaFree(reinterpret_cast<void*>(d_origins));
    cudaFree(reinterpret_cast<void*>(d_dirs));
    cudaFree(reinterpret_cast<void*>(d_escaped));
    cudaFree(reinterpret_cast<void*>(d_params));

    return h_escaped;
}

} // namespace vpmr
