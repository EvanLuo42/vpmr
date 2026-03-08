#include <gtest/gtest.h>
#include "test_helpers.h"
#include "utils/bounding_box.h"

using namespace vpmr;

TEST(AABB, FromMeshVerticesNoPadding) {
    auto m = test_helpers::make_cube();
    auto box = AABB::from_mesh_vertices(m.V);
    EXPECT_NEAR(box.min_pt(0), 0.0, 1e-12);
    EXPECT_NEAR(box.min_pt(1), 0.0, 1e-12);
    EXPECT_NEAR(box.min_pt(2), 0.0, 1e-12);
    EXPECT_NEAR(box.max_pt(0), 1.0, 1e-12);
    EXPECT_NEAR(box.max_pt(1), 1.0, 1e-12);
    EXPECT_NEAR(box.max_pt(2), 1.0, 1e-12);
}

TEST(AABB, FromMeshVerticesWithPadding) {
    auto m = test_helpers::make_cube();
    auto box = AABB::from_mesh_vertices(m.V, 0.5);
    EXPECT_NEAR(box.min_pt(0), -0.5, 1e-12);
    EXPECT_NEAR(box.max_pt(0), 1.5, 1e-12);
}

TEST(AABB, ContainsInside) {
    auto m = test_helpers::make_cube();
    auto box = AABB::from_mesh_vertices(m.V);
    EXPECT_TRUE(box.contains(Vec3d(0.5, 0.5, 0.5)));
}

TEST(AABB, ContainsOnBoundary) {
    auto m = test_helpers::make_cube();
    auto box = AABB::from_mesh_vertices(m.V);
    EXPECT_TRUE(box.contains(Vec3d(0.0, 0.0, 0.0)));
    EXPECT_TRUE(box.contains(Vec3d(1.0, 1.0, 1.0)));
}

TEST(AABB, ContainsOutside) {
    auto m = test_helpers::make_cube();
    auto box = AABB::from_mesh_vertices(m.V);
    EXPECT_FALSE(box.contains(Vec3d(1.5, 0.5, 0.5)));
    EXPECT_FALSE(box.contains(Vec3d(-0.1, 0.5, 0.5)));
}

TEST(AABB, SingleVertex) {
    MatXd V(1, 3);
    V << 5.0, 3.0, 1.0;
    auto box = AABB::from_mesh_vertices(V);
    EXPECT_TRUE(box.contains(Vec3d(5.0, 3.0, 1.0)));
    EXPECT_FALSE(box.contains(Vec3d(5.1, 3.0, 1.0)));
}
