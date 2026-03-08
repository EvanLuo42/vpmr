#include <gtest/gtest.h>
#include "test_helpers.h"
#include "orientation/orientation.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Orientation, ConsistentOrientationUnchanged) {
    // Quad with consistent orientation and positive orientation measure
    auto m = make_quad();
    set_measures(m, 1.0, 1.0, 0.0);

    MatXi F_before = m.F;
    adjust_orientation(m);

    // Orientation is already consistent and positive → no flips
    EXPECT_EQ(m.F.rows(), F_before.rows());
    for (int f = 0; f < m.F.rows(); ++f) {
        EXPECT_EQ(m.F(f, 0), F_before(f, 0));
        EXPECT_EQ(m.F(f, 1), F_before(f, 1));
        EXPECT_EQ(m.F(f, 2), F_before(f, 2));
    }
}

TEST(Orientation, NegativeOrientationFlips) {
    // Single triangle with negative orientation → should flip
    auto m = make_single_triangle();
    set_measures(m, 1.0, -1.0, 0.0);

    int v0 = m.F(0, 0), v1 = m.F(0, 1), v2 = m.F(0, 2);
    adjust_orientation(m);

    // After flip, v1 and v2 should be swapped
    EXPECT_EQ(m.F(0, 0), v0);
    EXPECT_EQ(m.F(0, 1), v2);
    EXPECT_EQ(m.F(0, 2), v1);
    // Orientation measure should be negated
    EXPECT_GT(m.orientation(0), 0);
}

TEST(Orientation, DuplicateFacesRemoved) {
    Mesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0;
    // Two identical faces (same vertex set, different winding)
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           0, 2, 1;
    set_measures(m, 1.0, 1.0, 0.0);

    adjust_orientation(m);

    // Both duplicates should be removed (exact opposite pairs)
    EXPECT_EQ(m.F.rows(), 0);
}

TEST(Orientation, EmptyMeshHandled) {
    Mesh m;
    m.V.resize(0, 3);
    m.F.resize(0, 3);
    // Should not crash
    adjust_orientation(m);
    EXPECT_EQ(m.F.rows(), 0);
}

TEST(Orientation, InvisibleFacesDontAffectDecision) {
    // Two faces in same patch: one visible with positive ori, one invisible with negative
    auto m = make_quad();
    m.visibility.resize(2);
    m.orientation.resize(2);
    m.openness.resize(2);
    m.visibility << 1.0, 0.0;  // face 0 visible, face 1 invisible
    m.orientation << 1.0, -1.0; // face 0 positive, face 1 negative
    m.openness << 0.0, 0.0;

    MatXi F_before = m.F;
    adjust_orientation(m);

    // Only visible face (positive orientation) contributes → no flip
    EXPECT_EQ(m.F.rows(), 2);
    // Faces should remain unchanged since weighted orientation is positive
    for (int f = 0; f < m.F.rows(); ++f) {
        EXPECT_EQ(m.F(f, 0), F_before(f, 0));
    }
}

TEST(Orientation, MultiPatchIndependent) {
    // Two disconnected triangles: one with positive ori, one with negative
    Mesh m;
    m.V.resize(6, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           10, 0, 0,
           11, 0, 0,
           10, 1, 0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           3, 4, 5;
    set_measures(m, 1.0, 0.0, 0.0);
    m.orientation(0) = 1.0;  // positive → keep
    m.orientation(1) = -1.0; // negative → flip

    adjust_orientation(m);

    // Face 0 should be unchanged
    EXPECT_EQ(m.F(0, 0), 0);
    EXPECT_EQ(m.F(0, 1), 1);
    EXPECT_EQ(m.F(0, 2), 2);

    // Face 1 should be flipped (v1, v2 swapped)
    EXPECT_EQ(m.F(1, 0), 3);
    EXPECT_EQ(m.F(1, 1), 5);
    EXPECT_EQ(m.F(1, 2), 4);
}
