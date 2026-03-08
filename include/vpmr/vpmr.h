#pragma once

#include "vpmr/config.h"
#include "vpmr/mesh.h"

namespace vpmr {

Mesh run_pipeline(Mesh input, const Config &config = Config{});

} // namespace vpmr
