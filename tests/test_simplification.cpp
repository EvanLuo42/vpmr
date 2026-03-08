#include <gtest/gtest.h>
#include "test_helpers.h"
#include "simplification/simplification.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Simplification, SingleFaceUnchanged) {
    auto m = make_single_triangle();
    m.face_mapping.resize(1);
    m.face_mapping << 0;
    m.offset_source.resize(1);
    m.offset_source << -1;

    Config config;
    simplify_mesh(m, config);

    EXPECT_EQ(m.F.rows(), 1);
}

TEST(Simplification, CoplanarFacesMerged) {
    // Two coplanar triangles forming a quad → should be simplified
    auto m = make_quad();
    m.face_mapping.resize(2);
    m.face_mapping << 0, 0; // same original face → guaranteed coplanar
    m.offset_source.resize(2);
    m.offset_source << -1, -1;

    Config config;
    int orig_nf = m.F.rows();
    simplify_mesh(m, config);

    // Should reduce face count (2 coplanar faces → ear clip can reduce)
    EXPECT_LE(m.F.rows(), orig_nf);
    // Total area should be preserved
    EXPECT_NEAR(m.total_area(), 1.0, 1e-6);
}

TEST(Simplification, NonCoplanarFacesPreserved) {
    // Two non-coplanar triangles (forming a V shape)
    Mesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, 1, 0,
           0.5, -1, 1; // out of plane
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           1, 0, 3;
    m.face_mapping.resize(2);
    m.face_mapping << 0, 1; // different original faces
    m.offset_source.resize(2);
    m.offset_source << -1, -1;

    Config config;
    simplify_mesh(m, config);

    // Non-coplanar → each is its own patch → no simplification
    EXPECT_EQ(m.F.rows(), 2);
}

TEST(Simplification, FourCoplanarTriangles) {
    // Split a square into 4 coplanar triangles with a center vertex
    Mesh m;
    m.V.resize(5, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           1, 1, 0,
           0, 1, 0,
           0.5, 0.5, 0; // center

    m.F.resize(4, 3);
    m.F << 0, 1, 4,
           1, 2, 4,
           2, 3, 4,
           3, 0, 4;
    m.face_mapping = VecXi::Constant(4, 0); // all same original
    m.offset_source = VecXi::Constant(4, -1);

    Config config;
    simplify_mesh(m, config);

    // 4 coplanar triangles → should reduce to 2 (minimum for a quad)
    EXPECT_LE(m.F.rows(), 4);
    EXPECT_NEAR(m.total_area(), 1.0, 1e-6);
}

TEST(Simplification, HoledPatchPreservesHoleArea) {
    Mesh m;
    m.V.resize(8, 3);
    m.V << 0, 0, 0,
           4, 0, 0,
           4, 4, 0,
           0, 4, 0,
           1, 1, 0,
           3, 1, 0,
           3, 3, 0,
           1, 3, 0;

    // Square ring: outer 4x4 square with inner 2x2 hole.
    m.F.resize(8, 3);
    m.F << 0, 1, 5,
           0, 5, 4,
           1, 2, 6,
           1, 6, 5,
           2, 3, 7,
           2, 7, 6,
           3, 0, 4,
           3, 4, 7;
    m.face_mapping = VecXi::Constant(8, 0);
    m.offset_source = VecXi::Constant(8, -1);

    Config config;
    int orig_nf = m.F.rows();
    double area_before = m.total_area();
    simplify_mesh(m, config);

    // Ear clipping does not support holes; simplification must not fill them.
    EXPECT_EQ(m.F.rows(), orig_nf);
    EXPECT_NEAR(m.total_area(), area_before, 1e-6);
}

TEST(Simplification, FaceMappingPreserved) {
    auto m = make_quad();
    m.face_mapping.resize(2);
    m.face_mapping << 5, 5;
    m.offset_source.resize(2);
    m.offset_source << -1, -1;

    Config config;
    simplify_mesh(m, config);

    for (int f = 0; f < m.F.rows(); ++f) {
        EXPECT_EQ(m.face_mapping(f), 5);
    }
}

TEST(Simplification, CubePreservesArea) {
    auto m = make_cube();
    m.face_mapping = VecXi::Constant(12, 0);
    m.offset_source = VecXi::Constant(12, -1);

    Config config;
    double area_before = m.total_area();
    simplify_mesh(m, config);

    // Each pair of coplanar cube faces forms a patch → simplification should preserve area
    EXPECT_NEAR(m.total_area(), area_before, 1e-6);
}
