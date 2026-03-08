#include <gtest/gtest.h>
#include "test_helpers.h"
#include "offset/offset.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Offset, ClosedMeshNoOffset) {
    auto m = make_tetrahedron();
    set_measures(m, 1.0, 1.0, 0.0);
    int orig_nf = m.F.rows();

    Config config;
    config.boundary_ratio_threshold = 0.05;
    offset_open_surfaces(m, config);

    // Tetrahedron is closed (no boundary), no offset needed
    EXPECT_EQ(m.F.rows(), orig_nf);
}

TEST(Offset, HoleFillForNearlyClosedMesh) {
    // Create a mesh with a small hole (one face removed from a tetrahedron)
    Mesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, 1, 0,
           0.5, 0.5, 1;
    // Only 3 of the 4 tet faces → one boundary loop
    m.F.resize(3, 3);
    m.F << 0, 1, 3,
           1, 2, 3,
           2, 0, 3;
    set_measures(m, 1.0, 1.0, 1.0);

    Config config;
    // Set threshold high enough that this mesh falls into hole-fill path
    config.boundary_ratio_threshold = 1.0;
    offset_open_surfaces(m, config);

    // Should have added the missing bottom face
    EXPECT_GT(m.F.rows(), 3);
    // offset_source should be set for new faces
    EXPECT_EQ(m.offset_source.size(), m.F.rows());
}

TEST(Offset, OpenSurfaceCreatesOffsetShell) {
    auto m = make_open_surface();
    set_measures(m, 1.0, 1.0, 1.0); // all faces open

    Config config;
    config.boundary_ratio_threshold = 0.0; // force offset path
    config.offset_divisor = 1000.0;
    int orig_nf = m.F.rows();
    int orig_nv = m.V.rows();

    offset_open_surfaces(m, config);

    // Should have more faces (offset + side-stitching)
    EXPECT_GT(m.F.rows(), orig_nf);
    // Should have more vertices (offset copies)
    EXPECT_GT(m.V.rows(), orig_nv);
    // offset_source should exist
    EXPECT_EQ(m.offset_source.size(), m.F.rows());
}

TEST(Offset, OffsetSourceTracking) {
    auto m = make_single_triangle();
    set_measures(m, 1.0, 1.0, 1.0);

    Config config;
    config.boundary_ratio_threshold = 0.0;
    config.offset_divisor = 1000.0;

    offset_open_surfaces(m, config);

    // Original face should have offset_source = -1
    EXPECT_EQ(m.offset_source(0), -1);
    // At least one offset face should point back to face 0
    bool found = false;
    for (int f = 1; f < m.F.rows(); ++f) {
        if (m.offset_source(f) == 0) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(Offset, NoOpenFacesNoChange) {
    auto m = make_open_surface();
    set_measures(m, 1.0, 1.0, 0.0); // openness = 0 → no faces are open
    int orig_nf = m.F.rows();

    Config config;
    config.boundary_ratio_threshold = 0.0;
    config.epsilon_openness = 0.5;

    offset_open_surfaces(m, config);

    // No open faces → no offset created (only if boundary_ratio > threshold but no open faces)
    // Since boundary_ratio > 0 but no open patches, num_patches == 0, returns early
    EXPECT_EQ(m.F.rows(), orig_nf);
}
