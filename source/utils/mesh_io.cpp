#include "mesh_io.h"
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <filesystem>
#include <stdexcept>

namespace vpmr {

Mesh read_mesh(const std::string &path) {
    Mesh mesh;
    auto ext = std::filesystem::path(path).extension().string();
    if (ext == ".ply") {
        if (!igl::readPLY(path, mesh.V, mesh.F)) {
            throw std::runtime_error("Failed to read PLY: " + path);
        }
    } else if (ext == ".obj") {
        MatXd N;
        MatXi FN;
        if (!igl::readOBJ(path, mesh.V, mesh.TC, N, mesh.F, mesh.FTC, FN)) {
            throw std::runtime_error("Failed to read OBJ: " + path);
        }
    } else {
        throw std::runtime_error("Unsupported format: " + ext);
    }
    return mesh;
}

void write_mesh(const std::string &path, const Mesh &mesh) {
    auto ext = std::filesystem::path(path).extension().string();
    if (ext == ".ply") {
        if (!igl::writePLY(path, mesh.V, mesh.F)) {
            throw std::runtime_error("Failed to write PLY: " + path);
        }
    } else if (ext == ".obj") {
        if (!igl::writeOBJ(path, mesh.V, mesh.F)) {
            throw std::runtime_error("Failed to write OBJ: " + path);
        }
    } else {
        throw std::runtime_error("Unsupported format: " + ext);
    }
}

} // namespace vpmr
