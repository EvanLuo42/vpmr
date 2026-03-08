#include <optix.h>

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

extern "C" {
__constant__ vpmr::LaunchParams params;
}

// Simple PCG hash for GPU RNG
__device__ unsigned int pcg_hash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float rand_float(unsigned int &seed) {
    seed = pcg_hash(seed);
    return (float) seed / (float) 0xFFFFFFFFu;
}

__device__ float3 make_float3_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 make_float3_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 make_float3_scale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize3(float3 v) {
    float len = sqrtf(dot3(v, v));
    if (len > 1e-8f)
        return make_float3(v.x / len, v.y / len, v.z / len);
    return v;
}

__device__ bool bbox_contains(float3 p, float3 mn, float3 mx) {
    return p.x >= mn.x && p.x <= mx.x && p.y >= mn.y && p.y <= mx.y && p.z >= mn.z && p.z <= mx.z;
}

__device__ float3 random_hemisphere(float3 normal, unsigned int &seed) {
    float3 dir;
    for (int attempt = 0; attempt < 100; ++attempt) {
        dir.x = rand_float(seed) * 2.0f - 1.0f;
        dir.y = rand_float(seed) * 2.0f - 1.0f;
        dir.z = rand_float(seed) * 2.0f - 1.0f;
        float sq = dot3(dir, dir);
        if (sq > 1e-8f && sq <= 1.0f) {
            dir = normalize3(dir);
            if (dot3(dir, normal) < 0)
                dir = make_float3(-dir.x, -dir.y, -dir.z);
            return dir;
        }
    }
    return normal;
}

extern "C" __global__ void __raygen__trace() {
    uint3 idx = optixGetLaunchIndex();
    int ray_idx = idx.x;
    if (ray_idx >= params.num_rays)
        return;

    float3 origin = params.ray_origins[ray_idx];
    float3 direction = params.ray_directions[ray_idx];

    unsigned int seed = pcg_hash(static_cast<unsigned int>(ray_idx) * 1237u + 42u);
    bool did_escape = false;

    float3 cur_origin = origin;
    float3 cur_dir = direction;

    for (int bounce = 0; bounce <= params.N_b; ++bounce) {
        unsigned int p0 = 0;
        unsigned int p1 = __float_as_uint(1e20f);

        optixTrace(params.traversable, cur_origin, cur_dir, 0.0001f, 1e20f, 0.0f, OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1);

        int face_id = (int) p0;
        float t = __uint_as_float(p1);

        if (face_id == -1 || t < 0.0f) {
            did_escape = true;
            break;
        }

        float3 hit_pt = make_float3_add(cur_origin, make_float3_scale(cur_dir, t));

        if (!bbox_contains(hit_pt, params.bbox_min, params.bbox_max)) {
            did_escape = true;
            break;
        }

        // Reflect: pick random direction in hemisphere facing the ray's side
        float3 hit_n = params.face_normals[face_id];
        if (dot3(cur_dir, hit_n) > 0.0f)
            hit_n = make_float3(-hit_n.x, -hit_n.y, -hit_n.z);
        cur_origin = hit_pt;
        cur_dir = random_hemisphere(hit_n, seed);
    }

    params.escaped[ray_idx] = did_escape ? 1 : 0;
}

extern "C" __global__ void __closesthit__trace() {
    int prim_idx = optixGetPrimitiveIndex();
    float t = optixGetRayTmax();
    optixSetPayload_0((unsigned int) prim_idx);
    optixSetPayload_1(__float_as_uint(t));
}

extern "C" __global__ void __miss__trace() {
    optixSetPayload_0((unsigned int) -1);
    optixSetPayload_1(__float_as_uint(-1.0f));
}
