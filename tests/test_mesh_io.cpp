#include <gtest/gtest.h>
#include "test_helpers.h"
#include "utils/mesh_io.h"
#include <filesystem>
#include <fstream>

using namespace vpmr;
using namespace test_helpers;

class MeshIOTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_dir;

    void SetUp() override {
        tmp_dir = std::filesystem::temp_directory_path() / "vpmr_test_io";
        std::filesystem::create_directories(tmp_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(tmp_dir);
    }
};

TEST_F(MeshIOTest, WriteAndReadPLY) {
    auto m = make_cube();
    auto path = (tmp_dir / "test.ply").string();

    write_mesh(path, m);
    EXPECT_TRUE(std::filesystem::exists(path));

    auto loaded = read_mesh(path);
    EXPECT_EQ(loaded.V.rows(), m.V.rows());
    EXPECT_EQ(loaded.F.rows(), m.F.rows());
    EXPECT_EQ(loaded.V.cols(), 3);
    EXPECT_EQ(loaded.F.cols(), 3);

    for (int i = 0; i < m.V.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(loaded.V(i, j), m.V(i, j), 1e-6);
        }
    }
}

TEST_F(MeshIOTest, WriteAndReadOBJ) {
    auto m = make_single_triangle();
    auto path = (tmp_dir / "test.obj").string();

    write_mesh(path, m);
    EXPECT_TRUE(std::filesystem::exists(path));

    auto loaded = read_mesh(path);
    EXPECT_EQ(loaded.V.rows(), 3);
    EXPECT_EQ(loaded.F.rows(), 1);
}

TEST_F(MeshIOTest, UnsupportedFormatThrows) {
    auto m = make_single_triangle();
    auto path = (tmp_dir / "test.stl").string();
    EXPECT_THROW(write_mesh(path, m), std::runtime_error);
}

TEST_F(MeshIOTest, ReadNonexistentThrows) {
    EXPECT_ANY_THROW(read_mesh("/nonexistent/path/mesh.ply"));
}
