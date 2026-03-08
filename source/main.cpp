#include "vpmr/vpmr.h"
#include "utils/mesh_io.h"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: vpmr <input_mesh> <output_mesh>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    try {
        auto mesh = vpmr::read_mesh(input_path);
        std::cout << "Loaded: " << mesh.V.rows() << " vertices, " << mesh.F.rows() << " faces" << std::endl;

        vpmr::Config config;
        auto result = vpmr::run_pipeline(std::move(mesh), config);

        vpmr::write_mesh(output_path, result);
        std::cout << "Saved: " << output_path << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
