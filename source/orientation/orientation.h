#pragma once

#include "vpmr/mesh.h"

namespace vpmr
{

void adjust_orientation(Mesh& mesh, MatXi* per_face_cells = nullptr);

} // namespace vpmr
