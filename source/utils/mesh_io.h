#pragma once

#include "vpmr/mesh.h"
#include <string>

namespace vpmr {

Mesh read_mesh(const std::string &path);
void write_mesh(const std::string &path, const Mesh &mesh);

} // namespace vpmr
