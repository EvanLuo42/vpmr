#include <gtest/gtest.h>
#include "test_helpers.h"
#include "partition/partition.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Partition, SingleTriangle)
{
    auto m = make_single_triangle();
    set_measures(m, 1.0, 1.0, 0.0);
    int original_face_count = m.F.rows();

    auto result = partition_space(m, original_face_count);

    // A single triangle (coplanar points) may not produce Delaunay tets
    // The partition should still return a valid (possibly empty) result
    EXPECT_GE(result.num_cells, 1); // at least the unbounded cell
    EXPECT_EQ(result.per_patch_cells.rows(), result.mesh.F.rows());
    if (result.mesh.F.rows() > 0)
    {
        EXPECT_EQ(result.per_patch_cells.cols(), 2);
    }
}

TEST(Partition, ClosedTetrahedron)
{
    auto m = make_tetrahedron();
    set_measures(m, 1.0, 1.0, 0.0);
    int original_face_count = m.F.rows();

    auto result = partition_space(m, original_face_count);

    EXPECT_GT(result.mesh.F.rows(), 0);
    EXPECT_GT(result.num_cells, 1); // at least unbounded + 1 interior
}

TEST(Partition, FaceMappingValid)
{
    auto m = make_single_triangle();
    set_measures(m, 1.0, 1.0, 0.0);
    int original_face_count = m.F.rows();

    auto result = partition_space(m, original_face_count);

    EXPECT_EQ(result.mesh.face_mapping.size(), result.mesh.F.rows());
    // face_mapping values: >= 0 means original face, < 0 means extra
    for (int f = 0; f < result.mesh.F.rows(); ++f)
    {
        if (result.mesh.face_mapping(f) >= 0)
        {
            EXPECT_LT(result.mesh.face_mapping(f), original_face_count);
        }
    }
}

TEST(Partition, PerPatchCellsValid)
{
    auto m = make_single_triangle();
    set_measures(m, 1.0, 1.0, 0.0);

    auto result = partition_space(m, m.F.rows());

    for (int f = 0; f < result.mesh.F.rows(); ++f)
    {
        int lc = result.per_patch_cells(f, 0);
        int rc = result.per_patch_cells(f, 1);
        EXPECT_GE(lc, 0);
        EXPECT_LT(lc, result.num_cells);
        EXPECT_GE(rc, 0);
        EXPECT_LT(rc, result.num_cells);
        EXPECT_NE(lc, rc); // cells on either side should differ
    }
}
