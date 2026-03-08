#pragma once

#include "vpmr/config.h"
#include "vpmr/mesh.h"
#include "../partition/partition.h"

namespace vpmr {

Mesh extract_interface(const PartitionResult &partition, const Config &config);

} // namespace vpmr
