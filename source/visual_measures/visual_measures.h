#pragma once

#include "vpmr/config.h"
#include "vpmr/mesh.h"

namespace vpmr {

void compute_visual_measures(Mesh &target_mesh, const Mesh &scene_mesh, const Config &config);
void compute_visual_measures(Mesh &mesh, const Config &config);

} // namespace vpmr
