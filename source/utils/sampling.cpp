#include "sampling.h"
#include <cmath>
#include <random>

namespace vpmr
{

std::vector<SamplePoint> sample_faces(const Mesh& mesh, int N_min, int N_total)
{
    const int nf = static_cast<int>(mesh.F.rows());
    const double total_area = mesh.total_area();

    std::vector<int> samples_per_face(nf);
    for (int i = 0; i < nf; ++i)
    {
        const double a = mesh.face_area(i);
        const int ns = std::max(N_min, static_cast<int>(std::ceil(a / total_area * N_total)));
        samples_per_face[i] = ns;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution dist(0.0, 1.0);

    std::vector<SamplePoint> samples;
    for (int f = 0; f < nf; ++f)
    {
        Vec3d v0 = mesh.V.row(mesh.F(f, 0));
        Vec3d v1 = mesh.V.row(mesh.F(f, 1));
        Vec3d v2 = mesh.V.row(mesh.F(f, 2));
        const Vec3d n = mesh.face_normal(f);

        for (int s = 0; s < samples_per_face[f]; ++s)
        {
            double r1 = dist(rng);
            double r2 = dist(rng);
            if (r1 + r2 > 1.0)
            {
                r1 = 1.0 - r1;
                r2 = 1.0 - r2;
            }
            const Vec3d p = v0 + r1 * (v1 - v0) + r2 * (v2 - v0);
            samples.push_back({f, p, n});
        }
    }
    return samples;
}

} // namespace vpmr
