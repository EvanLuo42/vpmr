#include <gtest/gtest.h>
#include "test_helpers.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Mesh, BboxDiagonalSingleTriangle) {
    auto m = make_single_triangle();
    // Bbox: (0,0,0) to (1,1,0), diagonal = sqrt(1+1+0) = sqrt(2)
    EXPECT_NEAR(m.bbox_diagonal(), std::sqrt(2.0), 1e-12);
}

TEST(Mesh, BboxDiagonalCube) {
    auto m = make_cube();
    // Bbox: (0,0,0) to (1,1,1), diagonal = sqrt(3)
    EXPECT_NEAR(m.bbox_diagonal(), std::sqrt(3.0), 1e-12);
}

TEST(Mesh, FaceNormalXYPlane) {
    auto m = make_single_triangle();
    Vec3d n = m.face_normal(0);
    // (1,0,0)-(0,0,0) cross (0,1,0)-(0,0,0) = (1,0,0) cross (0,1,0) = (0,0,1)
    EXPECT_NEAR(n(0), 0.0, 1e-12);
    EXPECT_NEAR(n(1), 0.0, 1e-12);
    EXPECT_NEAR(n(2), 1.0, 1e-12);
}

TEST(Mesh, FaceNormalUnitLength) {
    auto m = make_cube();
    for (int f = 0; f < m.F.rows(); ++f) {
        Vec3d n = m.face_normal(f);
        EXPECT_NEAR(n.norm(), 1.0, 1e-12);
    }
}

TEST(Mesh, FaceNormalDegenerateTriangle) {
    Mesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           2, 0, 0; // collinear
    m.F.resize(1, 3);
    m.F << 0, 1, 2;
    Vec3d n = m.face_normal(0);
    EXPECT_NEAR(n.norm(), 0.0, 1e-12);
}

TEST(Mesh, FaceAreaSingleTriangle) {
    auto m = make_single_triangle();
    // Right triangle with legs 1, area = 0.5
    EXPECT_NEAR(m.face_area(0), 0.5, 1e-12);
}

TEST(Mesh, FaceAreaScaled) {
    Mesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           2, 0, 0,
           0, 2, 0;
    m.F.resize(1, 3);
    m.F << 0, 1, 2;
    EXPECT_NEAR(m.face_area(0), 2.0, 1e-12);
}

TEST(Mesh, TotalAreaQuad) {
    auto m = make_quad();
    // Two right triangles, each area 0.5 → total 1.0
    EXPECT_NEAR(m.total_area(), 1.0, 1e-12);
}

TEST(Mesh, TotalAreaCube) {
    auto m = make_cube();
    // 6 faces of unit cube, each area 1.0 → total 6.0
    EXPECT_NEAR(m.total_area(), 6.0, 1e-12);
}

TEST(Mesh, FaceCentroid) {
    auto m = make_single_triangle();
    Vec3d c = m.face_centroid(0);
    EXPECT_NEAR(c(0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(c(1), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(c(2), 0.0, 1e-12);
}

TEST(Mesh, EmptyMeshTotalArea) {
    Mesh m;
    m.V.resize(0, 3);
    m.F.resize(0, 3);
    EXPECT_NEAR(m.total_area(), 0.0, 1e-12);
}
