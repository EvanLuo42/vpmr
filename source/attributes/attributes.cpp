#include "attributes.h"
#include <igl/point_mesh_squared_distance.h>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace vpmr {

static void barycentric_coords(const Vec3d &p, const Vec3d &a, const Vec3d &b, const Vec3d &c, double &u, double &v,
                                double &w) {
    Vec3d v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1);
    double d20 = v2.dot(v0), d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-15) {
        u = v = w = 1.0 / 3.0;
        return;
    }
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0 - v - w;
}

void recover_attributes(Mesh &result, const Mesh &input) {
    int nf = result.F.rows();
    bool has_tc = input.TC.rows() > 0 && input.FTC.rows() > 0;
    bool has_vc = input.VC.rows() > 0;

    if (!has_tc && !has_vc)
        return;

    if (has_tc) {
        result.TC.resize(nf * 3, 2);
        result.FTC.resize(nf, 3);
    }
    if (has_vc) {
        result.VC.resize(result.V.rows(), input.VC.cols());
        result.VC.setZero();
        std::vector<int> vc_count(result.V.rows(), 0);
    }

    std::vector<bool> processed(nf, false);

    for (int f = 0; f < nf; ++f) {
        int src = -1;

        // Check offset_source first (paper §3.8: offset face → copy from original)
        if (result.offset_source.size() > 0 && result.offset_source(f) >= 0) {
            src = result.offset_source(f);
        }
        // Then check face_mapping (inherited face)
        else if (result.face_mapping.size() > 0 && result.face_mapping(f) >= 0) {
            src = result.face_mapping(f);
        }

        if (src >= 0 && src < input.F.rows()) {
            if (has_tc && src < input.FTC.rows()) {
                // Barycentric interpolation from input face
                Vec3d a = input.V.row(input.F(src, 0));
                Vec3d b = input.V.row(input.F(src, 1));
                Vec3d c = input.V.row(input.F(src, 2));

                for (int j = 0; j < 3; ++j) {
                    Vec3d p = result.V.row(result.F(f, j));
                    double u, v, w;
                    barycentric_coords(p, a, b, c, u, v, w);

                    Eigen::Vector2d tc0 = input.TC.row(input.FTC(src, 0));
                    Eigen::Vector2d tc1 = input.TC.row(input.FTC(src, 1));
                    Eigen::Vector2d tc2 = input.TC.row(input.FTC(src, 2));

                    result.TC.row(f * 3 + j) = u * tc0 + v * tc1 + w * tc2;
                    result.FTC(f, j) = f * 3 + j;
                }
            }
            processed[f] = true;
        }
    }

    // Flood-fill for extra faces (no mapping or offset source)
    if (has_tc) {
        using Edge = std::pair<int, int>;
        std::map<Edge, std::vector<int>> edge_faces;
        for (int f = 0; f < nf; ++f) {
            for (int e = 0; e < 3; ++e) {
                int a = result.F(f, e);
                int b = result.F(f, (e + 1) % 3);
                Edge edge = a < b ? Edge{a, b} : Edge{b, a};
                edge_faces[edge].push_back(f);
            }
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (int f = 0; f < nf; ++f) {
                if (processed[f])
                    continue;

                // Average TC from processed neighbors
                int count = 0;
                Eigen::MatrixXd avg_tc = Eigen::MatrixXd::Zero(3, 2);

                for (int e = 0; e < 3; ++e) {
                    int a = result.F(f, e);
                    int b = result.F(f, (e + 1) % 3);
                    Edge edge = a < b ? Edge{a, b} : Edge{b, a};
                    for (int nb : edge_faces[edge]) {
                        if (nb != f && processed[nb]) {
                            for (int j = 0; j < 3; ++j) {
                                avg_tc.row(j) += result.TC.row(nb * 3 + j);
                            }
                            ++count;
                        }
                    }
                }

                if (count > 0) {
                    for (int j = 0; j < 3; ++j) {
                        result.TC.row(f * 3 + j) = avg_tc.row(j) / count;
                        result.FTC(f, j) = f * 3 + j;
                    }
                    processed[f] = true;
                    changed = true;
                }
            }
        }
    }
}

} // namespace vpmr
