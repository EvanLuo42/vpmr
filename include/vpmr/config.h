#pragma once

namespace vpmr {

struct Config {
    int N_min = 5;
    int N_total = 20000000;
    int N_d = 5;
    int N_b = 10;

    double epsilon_openness = 0.5;
    double offset_divisor = 20000.0;
    double boundary_ratio_threshold = 0.05;

    double l_extended_divisor = 1000.0;
};

} // namespace vpmr
