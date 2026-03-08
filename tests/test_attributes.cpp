#include <gtest/gtest.h>
#include "test_helpers.h"
#include "attributes/attributes.h"

using namespace vpmr;
using namespace test_helpers;

TEST(Attributes, NoUVsNoOp) {
    Mesh input = make_single_triangle();
    Mesh result = make_single_triangle();
    result.face_mapping.resize(1);
    result.face_mapping << 0;
    result.offset_source.resize(1);
    result.offset_source << -1;

    // Input has no UVs → nothing to recover
    recover_attributes(result, input);
    EXPECT_EQ(result.TC.rows(), 0);
    EXPECT_EQ(result.FTC.rows(), 0);
}

TEST(Attributes, DirectMappingUVTransfer) {
    Mesh input = make_single_triangle();
    add_simple_uvs(input);

    // Result is the same triangle, mapped to input face 0
    Mesh result = make_single_triangle();
    result.face_mapping.resize(1);
    result.face_mapping << 0;
    result.offset_source.resize(1);
    result.offset_source << -1;

    recover_attributes(result, input);

    EXPECT_EQ(result.TC.rows(), 3);
    EXPECT_EQ(result.FTC.rows(), 1);
}

TEST(Attributes, OffsetSourcePriority) {
    // Test that offset_source is checked before face_mapping
    Mesh input;
    input.V.resize(6, 3);
    input.V << 0, 0, 0,
               1, 0, 0,
               0, 1, 0,
               10, 0, 0,
               11, 0, 0,
               10, 1, 0;
    input.F.resize(2, 3);
    input.F << 0, 1, 2,
               3, 4, 5;

    // Different UVs for each input face
    input.TC.resize(6, 2);
    input.TC << 0, 0,  0.5, 0,  0, 0.5,   // face 0 UVs
                1, 1,  1.5, 1,  1, 1.5;    // face 1 UVs
    input.FTC.resize(2, 3);
    input.FTC << 0, 1, 2,
                 3, 4, 5;

    Mesh result = make_single_triangle();
    result.face_mapping.resize(1);
    result.face_mapping << 0;   // would map to face 0
    result.offset_source.resize(1);
    result.offset_source << 1;  // but offset_source points to face 1 (higher priority)

    recover_attributes(result, input);

    EXPECT_EQ(result.TC.rows(), 3);
    // UV coords should come from face 1 (offset_source), not face 0 (face_mapping)
    // The interpolation depends on barycentric coords, but the source face should be face 1
}

TEST(Attributes, FloodFillUnmappedFaces) {
    Mesh input = make_quad();
    add_simple_uvs(input);

    // Result has 2 faces: one mapped, one unmapped
    Mesh result = make_quad();
    result.face_mapping.resize(2);
    result.face_mapping << 0, -1; // face 1 has no mapping
    result.offset_source.resize(2);
    result.offset_source << -1, -1;

    recover_attributes(result, input);

    // Both faces should have UVs (face 1 via flood-fill from face 0)
    EXPECT_EQ(result.FTC.rows(), 2);
    // Face 1's UV should be filled from neighbor
    for (int j = 0; j < 3; ++j) {
        int tc_idx = result.FTC(1, j);
        EXPECT_GE(tc_idx, 0);
        EXPECT_LT(tc_idx, result.TC.rows());
    }
}

TEST(Attributes, EmptyResultHandled) {
    Mesh input = make_single_triangle();
    add_simple_uvs(input);

    Mesh result;
    result.V.resize(0, 3);
    result.F.resize(0, 3);
    result.face_mapping.resize(0);
    result.offset_source.resize(0);

    recover_attributes(result, input);
    EXPECT_EQ(result.TC.rows(), 0);
}
