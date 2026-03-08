#include <gtest/gtest.h>
#include <algorithm>
#include <map>
#include "test_helpers.h"
#include "topology/topology.h"

using namespace vpmr;
using namespace test_helpers;

using Edge = std::pair<int, int>;

static std::map<Edge, int> edge_incidence(const Mesh &mesh) {
    std::map<Edge, int> counts;
    for (int f = 0; f < mesh.F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int a = mesh.F(f, e);
            int b = mesh.F(f, (e + 1) % 3);
            counts[a < b ? Edge{a, b} : Edge{b, a}]++;
        }
    }
    return counts;
}

static int boundary_edge_count(const Mesh &mesh) {
    int boundary = 0;
    for (auto &[edge, count] : edge_incidence(mesh)) {
        if (count == 1)
            ++boundary;
    }
    return boundary;
}

static int max_edge_incidence(const Mesh &mesh) {
    int max_count = 0;
    for (auto &[edge, count] : edge_incidence(mesh))
        max_count = std::max(max_count, count);
    return max_count;
}

TEST(Topology, ManifoldMeshUnchanged) {
    auto m = make_tetrahedron();
    int orig_nv = m.V.rows();
    int orig_nf = m.F.rows();

    fix_topology(m);

    EXPECT_EQ(m.V.rows(), orig_nv);
    EXPECT_EQ(m.F.rows(), orig_nf);
}

TEST(Topology, NonManifoldEdgeSplit) {
    // Create a non-manifold edge: 3 faces sharing edge (0,1)
    Mesh m;
    m.V.resize(5, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, 1, 0,
           0.5, -1, 0,
           0.5, 0, 1;

    m.F.resize(3, 3);
    m.F << 0, 1, 2,  // face in XY plane
           0, 1, 3,  // face in XY plane (other side)
           0, 1, 4;  // face going up

    int orig_nf = m.F.rows();
    fix_topology(m);

    // Splitting removes the non-manifold edge, even when watertightness is
    // impossible because the edge incidence is odd.
    EXPECT_GT(m.F.rows(), orig_nf);
    EXPECT_GT(m.V.rows(), 5);
    EXPECT_LE(max_edge_incidence(m), 2);
}

TEST(Topology, NonManifoldVertexSplit) {
    // Two triangles sharing only a single vertex (bowtie) → non-manifold vertex
    Mesh m;
    m.V.resize(5, 3);
    m.V << 0, 0, 0,   // shared vertex
           1, 0, 0,
           0, 1, 0,
           -1, 0, 0,
           0, -1, 0;

    m.F.resize(2, 3);
    m.F << 0, 1, 2,   // triangle 1
           0, 3, 4;   // triangle 2 (only shares vertex 0)

    fix_topology(m);

    // Vertex 0 should be duplicated → 6 vertices
    EXPECT_EQ(m.V.rows(), 6);
    EXPECT_EQ(m.F.rows(), 2);

    // The two faces should not share any vertices anymore
    std::set<int> face0_verts = {m.F(0, 0), m.F(0, 1), m.F(0, 2)};
    std::set<int> face1_verts = {m.F(1, 0), m.F(1, 1), m.F(1, 2)};
    for (int v : face1_verts) {
        EXPECT_EQ(face0_verts.count(v), 0);
    }
}

TEST(Topology, DuplicateFacesCollapsed) {
    Mesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0;
    m.F.resize(3, 3);
    m.F << 0, 1, 2,
           2, 1, 0,
           0, 1, 2;
    m.face_mapping.resize(3);
    m.face_mapping << 4, 5, 6;
    m.offset_source = VecXi::Constant(3, -1);

    fix_topology(m);

    EXPECT_EQ(m.F.rows(), 1);
    EXPECT_EQ(m.face_mapping.size(), 1);
}

TEST(Topology, EmptyMeshHandled) {
    Mesh m;
    m.V.resize(0, 3);
    m.F.resize(0, 3);
    fix_topology(m);
    EXPECT_EQ(m.F.rows(), 0);
}

TEST(Topology, SingleFaceUnchanged) {
    auto m = make_single_triangle();
    fix_topology(m);
    EXPECT_EQ(m.V.rows(), 3);
    EXPECT_EQ(m.F.rows(), 1);
}

TEST(Topology, ManifoldEdgeTwoFaces) {
    // Two faces sharing an edge properly (manifold) → no change
    auto m = make_open_surface();
    int orig_nv = m.V.rows();
    fix_topology(m);
    EXPECT_EQ(m.V.rows(), orig_nv);
}

TEST(Topology, MultipleNonManifoldEdges) {
    // 4 faces all sharing edge (0,1)
    Mesh m;
    m.V.resize(6, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, 1, 0,
           0.5, -1, 0,
           0.5, 0, 1,
           0.5, 0, -1;

    m.F.resize(4, 3);
    m.F << 0, 1, 2,
           0, 1, 3,
           0, 1, 4,
           0, 1, 5;

    fix_topology(m);

    // Repeated edge splitting should remove every edge with >2 incident faces.
    EXPECT_GT(m.V.rows(), 6);
    EXPECT_GT(m.F.rows(), 4);
    EXPECT_LE(max_edge_incidence(m), 2);
}

TEST(Topology, ClosedNonManifoldEdgePreservesWatertightness) {
    // Two tetrahedra sharing only edge (0,1): closed surface with one
    // non-manifold edge of even incidence.
    Mesh m;
    m.V.resize(6, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0.5, 1, 0,
           0.5, 0.2, 1,
           0.5, -1, 0,
           0.5, -0.2, -1;

    m.F.resize(8, 3);
    m.F << 0, 2, 1,
           0, 1, 3,
           1, 2, 3,
           2, 0, 3,
           0, 4, 1,
           0, 1, 5,
           1, 4, 5,
           4, 0, 5;

    EXPECT_EQ(boundary_edge_count(m), 0);
    EXPECT_GT(max_edge_incidence(m), 2);

    fix_topology(m);

    EXPECT_EQ(boundary_edge_count(m), 0);
    EXPECT_LE(max_edge_incidence(m), 2);
}
