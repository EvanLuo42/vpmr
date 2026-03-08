#include "vpmr/mesh.h"

namespace vpmr {

double Mesh::bbox_diagonal() const {
    Vec3d mn = V.colwise().minCoeff();
    Vec3d mx = V.colwise().maxCoeff();
    return (mx - mn).norm();
}

double Mesh::total_area() const {
    double area = 0;
    for (int i = 0; i < F.rows(); ++i) {
        area += face_area(i);
    }
    return area;
}

Vec3d Mesh::face_normal(int f) const {
    Vec3d v0 = V.row(F(f, 0));
    Vec3d v1 = V.row(F(f, 1));
    Vec3d v2 = V.row(F(f, 2));
    Vec3d n = (v1 - v0).cross(v2 - v0);
    double len = n.norm();
    if (len > 0)
        n /= len;
    return n;
}

double Mesh::face_area(int f) const {
    Vec3d v0 = V.row(F(f, 0));
    Vec3d v1 = V.row(F(f, 1));
    Vec3d v2 = V.row(F(f, 2));
    return 0.5 * (v1 - v0).cross(v2 - v0).norm();
}

Vec3d Mesh::face_centroid(int f) const {
    return (V.row(F(f, 0)) + V.row(F(f, 1)) + V.row(F(f, 2))).transpose() / 3.0;
}

} // namespace vpmr
