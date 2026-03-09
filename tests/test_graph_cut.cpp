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

static PartitionResult make_two_cell_partition_with_extra(double mapped_visibility)
{
    PartitionResult pr;

    // One mapped face with area 1, plus one extra face with smaller area 0.25.
    pr.mesh.V.resize(6, 3);
    pr.mesh.V << 0, 0, 0,
                 2, 0, 0,
                 0, 1, 0,
                 0, 0, 1,
                 1, 0, 1,
                 0, 0.5, 1;

    pr.mesh.F.resize(2, 3);
    pr.mesh.F << 0, 1, 2,
                 3, 4, 5;

    pr.num_cells = 2;
    pr.per_patch_cells.resize(2, 2);
    pr.per_patch_cells << 0, 1,
                         0, 1;

    pr.mesh.face_mapping.resize(2);
    pr.mesh.face_mapping << 0, -1;

    pr.mesh.offset_source.resize(2);
    pr.mesh.offset_source << -1, -1;

    pr.mesh.visibility.resize(2);
    pr.mesh.visibility << mapped_visibility, 0.0;

    pr.mesh.orientation.resize(2);
    pr.mesh.orientation << 1.0, 0.0;

    return pr;
}

static PartitionResult make_three_cell_partition_with_invisible_bridge()
{
    PartitionResult pr;

    // Cell 0 is exterior. A visible mapped face forces cell 1 interior.
    // Cell 2 is connected to cell 1 only through an invisible mapped face,
    // and to cell 0 through a smaller extra face. Without smoothness on the
    // invisible mapped face, cell 2 should stay exterior.
    pr.mesh.V.resize(9, 3);
    pr.mesh.V << 0, 0, 0,
                 2, 0, 0,
                 0, 1, 0,
                 0, 0, 1,
                 2, 0, 1,
                 0, 1, 1,
                 0, 0, 2,
                 1, 0, 2,
                 0, 0.5, 2;

    pr.mesh.F.resize(3, 3);
    pr.mesh.F << 0, 1, 2,
                 3, 4, 5,
                 6, 7, 8;

    pr.num_cells = 3;
    pr.per_patch_cells.resize(3, 2);
    pr.per_patch_cells << 0, 1,
                         1, 2,
                         0, 2;

    pr.mesh.face_mapping.resize(3);
    pr.mesh.face_mapping << 0, 1, -1;

    pr.mesh.offset_source.resize(3);
    pr.mesh.offset_source << -1, -1, -1;

    pr.mesh.visibility.resize(3);
    pr.mesh.visibility << 1.0, 0.0, 0.0;

    pr.mesh.orientation.resize(3);
    pr.mesh.orientation << 1.0, 1.0, 0.0;

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

    // ppc(f,0) = cell 0 = exterior → lc_interior = false → normal already
    // points outward (toward exterior) → faces keep original winding
    for (int f = 0; f < result.F.rows(); ++f)
    {
        EXPECT_EQ(result.F(f, 0), pr.mesh.F(f, 0));
        EXPECT_EQ(result.F(f, 1), pr.mesh.F(f, 1));
        EXPECT_EQ(result.F(f, 2), pr.mesh.F(f, 2));
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

TEST(GraphCut, VisibleFaceCanBeatSmallerExtraFace)
{
    auto pr = make_two_cell_partition_with_extra(0.51);

    Config config;
    Mesh result = extract_interface(pr, config);

    // The visible mapped face contributes unary cost 1.0, while the extra face
    // only contributes pairwise cost 0.25. The cut should therefore keep the
    // cell interior and expose both separating faces.
    EXPECT_EQ(result.F.rows(), 2);
    EXPECT_EQ(result.face_mapping(0), 0);
    EXPECT_EQ(result.face_mapping(1), -1);
}

TEST(GraphCut, InvisibleMappedFaceConnectsInterior)
{
    auto pr = make_three_cell_partition_with_invisible_bridge();

    Config config;
    Mesh result = extract_interface(pr, config);

    // The invisible mapped face now adds a smoothness edge, connecting cell 2
    // to cell 1 (interior). Since the smoothness to cell 1 (0.8) exceeds the
    // smoothness to cell 0 (0.2), cell 2 becomes interior. The interface is:
    //   face 0 (cell0-cell1, visible mapped) + face 2 (cell0-cell2, extra).
    EXPECT_EQ(result.F.rows(), 2);
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
