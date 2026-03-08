#include <gtest/gtest.h>
#include "test_helpers.h"
#include "partition/bsp.h"

using namespace vpmr;

TEST(BSPTree, InitDelaunay) {
    // Initialize BSP from a few points
    MatXd V(4, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;

    BSPTree bsp;
    bsp.init_delaunay(V);

    // Should have created cells from Delaunay tetrahedralization
    EXPECT_GT((int)bsp.cells.size(), 0);
    EXPECT_EQ((int)bsp.vertices.size(), (int)bsp.vertices_d.size());
}

TEST(BSPTree, InsertFace) {
    MatXd V(5, 3);
    V << 0, 0, 0,
         2, 0, 0,
         0, 2, 0,
         0, 0, 2,
         1, 1, 1;

    BSPTree bsp;
    bsp.init_delaunay(V);
    int cells_before = (int)bsp.cells.size();

    // Insert a face that may split some cells
    bsp.insert_face(0, 0, 1, 2);

    // After insertion, may have more cells (from splitting)
    EXPECT_GE((int)bsp.cells.size(), cells_before);
}

TEST(BSPTree, ExtractPartition) {
    MatXd V(4, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;

    BSPTree bsp;
    bsp.init_delaunay(V);
    bsp.insert_face(0, 0, 1, 2);

    MatXd V_out;
    MatXi F_out, per_face_cells;
    VecXi face_labels;
    int num_cells;

    bsp.extract_partition(V_out, F_out, per_face_cells, face_labels, num_cells);

    EXPECT_GT(V_out.rows(), 0);
    EXPECT_GT(F_out.rows(), 0);
    EXPECT_EQ(F_out.cols(), 3);
    EXPECT_EQ(per_face_cells.rows(), F_out.rows());
    EXPECT_EQ(per_face_cells.cols(), 2);
    EXPECT_EQ(face_labels.size(), F_out.rows());
    EXPECT_GT(num_cells, 0);
}

TEST(BSPTree, CellsHaveValidBounds) {
    MatXd V(4, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;

    BSPTree bsp;
    bsp.init_delaunay(V);

    for (auto &cell : bsp.cells) {
        // Each cell should have at least 4 faces (tetrahedron minimum)
        EXPECT_GE((int)cell.faces.size(), 4);
    }
}

TEST(BSPTree, MultipleFaceInsertions) {
    MatXd V(5, 3);
    V << 0, 0, 0,
         2, 0, 0,
         0, 2, 0,
         0, 0, 2,
         1, 1, 1;

    BSPTree bsp;
    bsp.init_delaunay(V);

    // Insert multiple faces
    bsp.insert_face(0, 0, 1, 2);
    bsp.insert_face(1, 0, 1, 3);
    bsp.insert_face(2, 0, 2, 3);

    MatXd V_out;
    MatXi F_out, per_face_cells;
    VecXi face_labels;
    int num_cells;

    bsp.extract_partition(V_out, F_out, per_face_cells, face_labels, num_cells);

    // Should have valid output
    EXPECT_GT(F_out.rows(), 0);
    EXPECT_GT(num_cells, 0);

    // Each face should have two valid cell references
    for (int f = 0; f < F_out.rows(); ++f) {
        EXPECT_GE(per_face_cells(f, 0), 0);
        EXPECT_LT(per_face_cells(f, 0), num_cells);
        EXPECT_GE(per_face_cells(f, 1), 0);
        EXPECT_LT(per_face_cells(f, 1), num_cells);
    }
}
