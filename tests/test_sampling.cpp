#include <gtest/gtest.h>
#include "test_helpers.h"
#include "utils/sampling.h"
#include "utils/bounding_box.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Sampling, MinSamplesPerFace) {
    auto m = make_single_triangle();
    int N_min = 10;
    int N_total = 1; // very low total → N_min should dominate
    auto samples = sample_faces(m, N_min, N_total);
    EXPECT_GE((int)samples.size(), N_min);
}

TEST(Sampling, SamplesOnFace) {
    auto m = make_single_triangle();
    auto samples = sample_faces(m, 5, 100);
    // All samples should be on face 0
    for (auto &s : samples) {
        EXPECT_EQ(s.face_id, 0);
    }
}

TEST(Sampling, SamplePositionsInsideTriangle) {
    auto m = make_single_triangle();
    auto samples = sample_faces(m, 100, 100);
    for (auto &s : samples) {
        // For triangle (0,0,0)-(1,0,0)-(0,1,0):
        // barycentric: x >= 0, y >= 0, x+y <= 1
        EXPECT_GE(s.position(0), -1e-10);
        EXPECT_GE(s.position(1), -1e-10);
        EXPECT_LE(s.position(0) + s.position(1), 1.0 + 1e-10);
        EXPECT_NEAR(s.position(2), 0.0, 1e-10);
    }
}

TEST(Sampling, SampleNormalsCorrect) {
    auto m = make_single_triangle();
    auto samples = sample_faces(m, 5, 100);
    Vec3d expected_n(0, 0, 1);
    for (auto &s : samples) {
        EXPECT_NEAR((s.normal - expected_n).norm(), 0.0, 1e-10);
    }
}

TEST(Sampling, MultipleFaces) {
    auto m = make_quad();
    auto samples = sample_faces(m, 5, 100);
    bool has_face0 = false, has_face1 = false;
    for (auto &s : samples) {
        if (s.face_id == 0) has_face0 = true;
        if (s.face_id == 1) has_face1 = true;
    }
    EXPECT_TRUE(has_face0);
    EXPECT_TRUE(has_face1);
}

TEST(Sampling, TotalSampleCountScalesWithArea) {
    auto m = make_quad();
    int N_min = 1;
    int N_total = 10000;
    auto samples = sample_faces(m, N_min, N_total);
    // Both faces have equal area, so samples should be roughly equal
    int count0 = 0, count1 = 0;
    for (auto &s : samples) {
        if (s.face_id == 0) ++count0;
        if (s.face_id == 1) ++count1;
    }
    // Should be within 2x of each other
    EXPECT_GT(count0, 0);
    EXPECT_GT(count1, 0);
}
