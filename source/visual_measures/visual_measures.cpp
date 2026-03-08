#include "visual_measures.h"
#include "optix_ray_tracer.h"
#include "../utils/sampling.h"
#include "../utils/bounding_box.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace vpmr
{

static Vec3d random_hemisphere_dir(const Vec3d& normal, std::mt19937& rng)
{
    std::uniform_real_distribution dist(-1.0, 1.0);
    Vec3d dir;
    do
    {
        dir = Vec3d(dist(rng), dist(rng), dist(rng));
    } while (dir.squaredNorm() > 1.0 || dir.squaredNorm() < 1e-8);
    dir.normalize();
    if (dir.dot(normal) < 0)
        dir = -dir;
    return dir;
}

void compute_visual_measures(Mesh& target_mesh, const Mesh& scene_mesh, const Config& config)
{
    int nf = target_mesh.F.rows();
    target_mesh.visibility.setZero(nf);
    target_mesh.orientation.setZero(nf);
    target_mesh.openness.setZero(nf);

    auto samples = sample_faces(target_mesh, config.N_min, config.N_total);
    int num_samples = (int)samples.size();
    std::cout << "[Visual Measures]   Samples: " << num_samples << std::endl;

    auto [min_pt, max_pt] = AABB::from_mesh_vertices(scene_mesh.V, scene_mesh.bbox_diagonal() * 0.01);

    OptixRayTracer tracer;
    tracer.build_accel(scene_mesh);
    tracer.upload_face_normals(scene_mesh);

    int N_d = config.N_d;
    int N_b = config.N_b;

    // Total rays: each sample x 2 hemispheres x N_d directions
    int total_rays = num_samples * 2 * N_d;
    std::cout << "[Visual Measures]   Total rays: " << total_rays << std::endl;

    // Pre-generate all initial ray origins and directions on CPU (parallel)
    std::vector<Vec3d> origins(total_rays);
    std::vector<Vec3d> directions(total_rays);

    {
        int num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        int chunk = (num_samples + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t)
        {
            int s_begin = t * chunk;
            int s_end = std::min(s_begin + chunk, num_samples);
            if (s_begin >= s_end) break;
            threads.emplace_back([&, s_begin, s_end, t]() {
                std::mt19937 rng(42u + t);
                for (int s = s_begin; s < s_end; ++s)
                {
                    Vec3d pos = samples[s].position;
                    Vec3d normal = samples[s].normal;
                    for (int sign = 0; sign < 2; ++sign)
                    {
                        Vec3d hemi_normal = sign == 0 ? normal : -normal;
                        for (int d = 0; d < N_d; ++d)
                        {
                            int idx = s * (2 * N_d) + sign * N_d + d;
                            origins[idx] = pos;
                            directions[idx] = random_hemisphere_dir(hemi_normal, rng);
                        }
                    }
                }
            });
        }
        for (auto &th : threads) th.join();
    }

    auto escaped = tracer.trace(origins, directions, N_b, min_pt, max_pt);

    size_t escaped_count = std::count(escaped.begin(), escaped.end(), true);
    std::cout << "[Visual Measures]   GPU trace complete — "
              << escaped_count << " / " << escaped.size() << " rays escaped ("
              << 100.0 * escaped_count / escaped.size() << "%)" << std::endl;

    // Aggregate results per face
    struct FaceAccum
    {
        double max_visibility = 0;
        double orient_pos_sum = 0;
        double orient_neg_sum = 0;
        double max_openness = 0;
        int sample_count = 0;
    };
    std::vector<FaceAccum> accum(nf);

    for (int s = 0; s < num_samples; ++s)
    {
        int f = samples[s].face_id;
        int valid_pos = 0, valid_neg = 0;

        for (int sign = 0; sign < 2; ++sign)
        {
            int valid_count = 0;
            for (int d = 0; d < N_d; ++d)
            {
                int idx = s * (2 * N_d) + sign * N_d + d;
                if (escaped[idx])
                    ++valid_count;
            }
            if (sign == 0)
                valid_pos = valid_count;
            else
                valid_neg = valid_count;
        }

        double sample_vis = static_cast<double>(std::max(valid_pos, valid_neg)) / N_d;
        accum[f].max_visibility = std::max(accum[f].max_visibility, sample_vis);

        accum[f].orient_pos_sum += valid_pos;
        accum[f].orient_neg_sum += valid_neg;

        if (sample_vis > 0.5)
        {
            int min_pn = std::min(valid_pos, valid_neg);
            int max_pn = std::max(valid_pos, valid_neg);
            double sample_open = 0.0;
            if (max_pn > 0)
            {
                sample_open = (static_cast<double>(min_pn) / max_pn)
                            * (valid_pos + valid_neg) / (2.0 * N_d);
            }
            accum[f].max_openness = std::max(accum[f].max_openness, sample_open);
        }
        accum[f].sample_count++;
    }

    for (int f = 0; f < nf; ++f)
    {
        auto& a = accum[f];
        target_mesh.visibility(f) = a.max_visibility;

        double total = a.orient_pos_sum + a.orient_neg_sum;
        if (total > 0)
        {
            target_mesh.orientation(f) = (a.orient_pos_sum - a.orient_neg_sum) / total;
        }

        target_mesh.openness(f) = a.max_openness;
    }
}

void compute_visual_measures(Mesh& mesh, const Config& config)
{
    compute_visual_measures(mesh, mesh, config);
}

} // namespace vpmr
