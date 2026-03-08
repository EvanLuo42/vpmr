#include <gtest/gtest.h>
#include "test_helpers.h"
#include "graph_cut/graph_cut.h"
#include "partition/partition.h"

using namespace vpmr;
using namespace test_helpers;

// Helper to create a simple PartitionResult manually
static PartitionResult make_simple_partition()
{
    PartitionResult pr;

    // 2 cells: cell 0 (exterior/unbounded), cell 1 (interior)
    // 2 faces separating them
    pr.mesh.V.resize(4, 3);
    pr.mesh.V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;

    pr.mesh.F.resize(2, 3);
    pr.mesh.F << 0, 1, 2, 0, 2, 3;

    pr.num_cells = 2;

    // ppc: face normals point from cell 0 (exterior) to cell 1 (interior)
    // With positive orientation, data term pushes lc=cell 0 → SINK, rc=cell 1 → SOURCE
    pr.per_patch_cells.resize(2, 2);
    pr.per_patch_cells << 0, 1, 0, 1;

    pr.mesh.face_mapping.resize(2);
    pr.mesh.face_mapping << 0, 1;

    pr.mesh.offset_source.resize(2);
    pr.mesh.offset_source << -1, -1;

    pr.mesh.visibility.resize(2);
    pr.mesh.visibility << 1.0, 1.0;

    pr.mesh.orientation.resize(2);
    pr.mesh.orientation << 1.0, 1.0;

    return pr;
}

TEST(GraphCut, AllVisiblePositiveOrientation)
{
    auto pr = make_simple_partition();
    Config config;

    Mesh result = extract_interface(pr, config);

    // Both faces are visible with positive orientation
    // Cell 0 forced to SINK (exterior), cell 1 should become interior
    // Both faces should be in the interface
    EXPECT_EQ(result.F.rows(), 2);
    EXPECT_EQ(result.V.rows(), pr.mesh.V.rows());
}

TEST(GraphCut, InterfaceFaceOrientation)
{
    auto pr = make_simple_partition();
    Config config;

    Mesh result = extract_interface(pr, config);

    // ppc(f,0) = cell 0 = exterior → lc_interior = false → faces get flipped
    // to orient normals outward (from interior to exterior)
    for (int f = 0; f < result.F.rows(); ++f)
    {
        EXPECT_EQ(result.F(f, 0), pr.mesh.F(f, 0));
        EXPECT_EQ(result.F(f, 1), pr.mesh.F(f, 2)); // flipped
        EXPECT_EQ(result.F(f, 2), pr.mesh.F(f, 1)); // flipped
    }
}

TEST(GraphCut, ExtraFacesSmoothnessOnly)
{
    // All faces are EXTRA (no mapping) → only smoothness edges
    PartitionResult pr;
    pr.mesh.V.resize(4, 3);
    pr.mesh.V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
    pr.mesh.F.resize(2, 3);
    pr.mesh.F << 0, 1, 2, 0, 2, 3;
    pr.num_cells = 2;
    pr.per_patch_cells.resize(2, 2);
    pr.per_patch_cells << 0, 1, 0, 1;
    pr.mesh.face_mapping.resize(2);
    pr.mesh.face_mapping << -1, -1;
    pr.mesh.offset_source.resize(2);
    pr.mesh.offset_source << -1, -1;
    pr.mesh.visibility = VecXd::Zero(2);
    pr.mesh.orientation = VecXd::Zero(2);

    Config config;
    Mesh result = extract_interface(pr, config);

    // With only smoothness edges and cell 0 forced to exterior,
    // no data term pushes cell 1 to interior → no interface faces.
    EXPECT_EQ(result.F.rows(), 0);
}

TEST(GraphCut, FaceMappingPreserved)
{
    auto pr = make_simple_partition();
    Config config;

    Mesh result = extract_interface(pr, config);

    EXPECT_EQ(result.F.rows(), 2);
    for (int f = 0; f < result.F.rows(); ++f)
    {
        EXPECT_GE(result.face_mapping(f), 0);
    }
}

TEST(GraphCut, EmptyPartition)
{
    PartitionResult pr;
    pr.mesh.V.resize(0, 3);
    pr.mesh.F.resize(0, 3);
    pr.num_cells = 1; // just the unbounded cell
    pr.per_patch_cells.resize(0, 2);
    pr.mesh.face_mapping.resize(0);
    pr.mesh.offset_source.resize(0);
    pr.mesh.visibility.resize(0);
    pr.mesh.orientation.resize(0);

    Config config;
    Mesh result = extract_interface(pr, config);
    EXPECT_EQ(result.F.rows(), 0);
}
