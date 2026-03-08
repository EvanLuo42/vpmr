#pragma once
#include "vpmr/mesh.h"
#include "vpmr/config.h"
#include "vpmr/types.h"

namespace test_helpers {

// Single triangle in XY plane: (0,0,0), (1,0,0), (0,1,0)
inline vpmr::Mesh make_single_triangle() {
    vpmr::Mesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0;
    m.F.resize(1, 3);
    m.F << 0, 1, 2;
    return m;
}

// Two adjacent triangles forming a quad in XY plane:
//  (0,0)-(1,0)-(1,1) and (0,0)-(1,1)-(0,1)
inline vpmr::Mesh make_quad() {
    vpmr::Mesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           1, 1, 0,
           0, 1, 0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           0, 2, 3;
    return m;
}

// Unit tetrahedron (closed mesh): 4 vertices, 4 faces
inline vpmr::Mesh make_tetrahedron() {
    vpmr::Mesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, std::sqrt(3.0) / 2.0, 0,
           0.5, std::sqrt(3.0) / 6.0, std::sqrt(6.0) / 3.0;
    // Outward-facing winding
    m.F.resize(4, 3);
    m.F << 0, 2, 1,   // bottom (normal -Z)
           0, 1, 3,   // front
           1, 2, 3,   // right
           2, 0, 3;   // left
    return m;
}

// Axis-aligned unit cube: 8 vertices, 12 faces (2 per face)
inline vpmr::Mesh make_cube() {
    vpmr::Mesh m;
    m.V.resize(8, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           1, 1, 0,
           0, 1, 0,
           0, 0, 1,
           1, 0, 1,
           1, 1, 1,
           0, 1, 1;
    // Outward-facing
    m.F.resize(12, 3);
    m.F << 0, 3, 2,   // bottom -Z
           0, 2, 1,
           4, 5, 6,   // top +Z
           4, 6, 7,
           0, 1, 5,   // front -Y
           0, 5, 4,
           2, 3, 7,   // back +Y
           2, 7, 6,
           0, 4, 7,   // left -X
           0, 7, 3,
           1, 2, 6,   // right +X
           1, 6, 5;
    return m;
}

// Open surface: two triangles sharing an edge, like an open book
inline vpmr::Mesh make_open_surface() {
    vpmr::Mesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, 1, 0,
           0.5, -1, 0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           1, 0, 3;
    return m;
}

// Set all visual measures to constant values
inline void set_measures(vpmr::Mesh &m, double vis, double ori, double open) {
    int nf = m.F.rows();
    m.visibility = vpmr::VecXd::Constant(nf, vis);
    m.orientation = vpmr::VecXd::Constant(nf, ori);
    m.openness = vpmr::VecXd::Constant(nf, open);
}

// Add simple UV coordinates (each face gets its own TC entries)
inline void add_simple_uvs(vpmr::Mesh &m) {
    int nf = m.F.rows();
    m.TC.resize(nf * 3, 2);
    m.FTC.resize(nf, 3);
    for (int f = 0; f < nf; ++f) {
        m.TC.row(f * 3 + 0) << 0.0, 0.0;
        m.TC.row(f * 3 + 1) << 1.0, 0.0;
        m.TC.row(f * 3 + 2) << 0.0, 1.0;
        m.FTC(f, 0) = f * 3 + 0;
        m.FTC(f, 1) = f * 3 + 1;
        m.FTC(f, 2) = f * 3 + 2;
    }
}

} // namespace test_helpers
