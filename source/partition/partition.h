#pragma once

#include "vpmr/mesh.h"

namespace vpmr
{

struct PartitionResult
{
    Mesh mesh;
    MatXi per_patch_cells; // per-face: (positive_cell, negative_cell)
    int num_cells;
};

PartitionResult partition_space(const Mesh& mesh, int original_face_count);

} // namespace vpmr
