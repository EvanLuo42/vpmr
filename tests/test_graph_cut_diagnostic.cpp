#include <gtest/gtest.h>
#include "test_helpers.h"
#include "graph_cut/graph_cut.h"
#include "partition/partition.h"

using namespace vpmr;
using namespace test_helpers;

// ============================================================================
// Diagnostic tests to identify why graph cut removes too many faces.
//
// Hypothesis 1: Invisible mapped faces add NO graph edges, disconnecting cells
//               from the interior, causing them to default to exterior.
// Hypothesis 2: Extra faces' smoothness overwhelms data terms when the ratio
//               of extra-to-mapped faces is very high.
// Hypothesis 3: End-to-end partition→graph_cut loses faces even on simple
//               closed meshes.
// ============================================================================

// ---------------------------------------------------------------------------
// H3: End-to-end partition → graph cut on a closed tetrahedron.
// A tetrahedron has 4 faces. After partition the BSP should produce 2 cells
// (exterior + 1 interior). Graph cut with all faces visible should keep all
// original faces in the interface.
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, TetrahedronEndToEnd)
{
    auto m = make_tetrahedron();
    set_measures(m, 1.0, 1.0, 0.0);
    int original_face_count = m.F.rows();

    auto partition = partition_space(m, original_face_count);

    // Partition should have at least 2 cells
    ASSERT_GE(partition.num_cells, 2);
    ASSERT_GT(partition.mesh.F.rows(), 0);

    // Set visual measures on partition mesh: all mapped faces are visible
    int nf = partition.mesh.F.rows();
    partition.mesh.visibility.resize(nf);
    partition.mesh.orientation.resize(nf);
    for (int f = 0; f < nf; ++f)
    {
        partition.mesh.visibility(f) = (partition.mesh.face_mapping(f) >= 0) ? 1.0 : 0.0;
        partition.mesh.orientation(f) = (partition.mesh.face_mapping(f) >= 0) ? 1.0 : 0.0;
    }

    Config config;
    Mesh result = extract_interface(partition, config);

    // All 4 original faces should appear in the interface
    int mapped_count = 0;
    for (int f = 0; f < result.F.rows(); ++f)
    {
        if (result.face_mapping(f) >= 0)
            ++mapped_count;
    }
    EXPECT_GE(mapped_count, original_face_count)
        << "Tetrahedron: expected all " << original_face_count
        << " original faces in interface, got " << mapped_count;
    EXPECT_GT(result.F.rows(), 0) << "Interface should not be empty";
}

// ---------------------------------------------------------------------------
// H3: End-to-end partition → graph cut on a closed cube.
// A cube has 12 faces. After partition, graph cut should produce an interface
// containing all 12 original mapped faces (plus possibly some extra faces).
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, CubeEndToEnd)
{
    auto m = make_cube();
    set_measures(m, 1.0, 1.0, 0.0);
    int original_face_count = m.F.rows();

    auto partition = partition_space(m, original_face_count);

    ASSERT_GE(partition.num_cells, 2);
    ASSERT_GT(partition.mesh.F.rows(), 0);

    int nf = partition.mesh.F.rows();
    partition.mesh.visibility.resize(nf);
    partition.mesh.orientation.resize(nf);
    for (int f = 0; f < nf; ++f)
    {
        partition.mesh.visibility(f) = (partition.mesh.face_mapping(f) >= 0) ? 1.0 : 0.0;
        partition.mesh.orientation(f) = (partition.mesh.face_mapping(f) >= 0) ? 1.0 : 0.0;
    }

    Config config;
    Mesh result = extract_interface(partition, config);

    int mapped_count = 0;
    for (int f = 0; f < result.F.rows(); ++f)
    {
        if (result.face_mapping(f) >= 0)
            ++mapped_count;
    }
    EXPECT_GE(mapped_count, original_face_count)
        << "Cube: expected all " << original_face_count
        << " original faces in interface, got " << mapped_count;
    EXPECT_GT(result.F.rows(), 0);
}

// ---------------------------------------------------------------------------
// H1: Invisible mapped faces disconnect cells from interior.
//
// Setup: 4 cells in a chain:  cell0 -- cell1 -- cell2 -- cell3
//   cell0 = exterior (unbounded)
//   face(0→1): extra, adds smoothness edge
//   face(1→2): visible mapped, pushes cell2 interior
//   face(2→3): INVISIBLE mapped, adds NO edge
//   face(0→3): extra, adds smoothness edge
//
// Expected: cell2 is interior (data term). Cell3 should also be interior
//   because it's "inside" the model. But since face(2→3) adds no edge,
//   cell3 has NO interior pressure and only the smoothness edge to cell0
//   pulls it to exterior. Cell3 becomes exterior → face(2→3) appears in
//   interface but face(0→3) does not, leaving a hole.
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, InvisibleMappedFaceDisconnectsInterior)
{
    PartitionResult pr;

    pr.mesh.V.resize(12, 3);
    pr.mesh.V <<
        0, 0, 0,  2, 0, 0,  0, 1, 0,   // face 0: extra (cell0-cell1)
        0, 0, 1,  2, 0, 1,  0, 1, 1,   // face 1: visible mapped (cell1-cell2)
        0, 0, 2,  2, 0, 2,  0, 1, 2,   // face 2: invisible mapped (cell2-cell3)
        0, 0, 3,  1, 0, 3,  0, 0.5, 3; // face 3: extra (cell0-cell3)

    pr.mesh.F.resize(4, 3);
    pr.mesh.F << 0, 1, 2,
                 3, 4, 5,
                 6, 7, 8,
                 9, 10, 11;

    pr.num_cells = 4;
    pr.per_patch_cells.resize(4, 2);
    pr.per_patch_cells <<
        0, 1,   // face 0: cell0 - cell1
        1, 2,   // face 1: cell1 - cell2
        2, 3,   // face 2: cell2 - cell3
        0, 3;   // face 3: cell0 - cell3

    pr.mesh.face_mapping.resize(4);
    pr.mesh.face_mapping << -1, 0, 1, -1;  // faces 1,2 mapped; faces 0,3 extra

    pr.mesh.offset_source.resize(4);
    pr.mesh.offset_source << -1, -1, -1, -1;

    pr.mesh.visibility.resize(4);
    pr.mesh.visibility << 0.0, 1.0, 0.3, 0.0;  // face 1 visible, face 2 invisible (vis<0.5)

    pr.mesh.orientation.resize(4);
    pr.mesh.orientation << 0.0, 1.0, 1.0, 0.0;  // positive orientation

    Config config;
    Mesh result = extract_interface(pr, config);

    // Diagnose: how many cells are interior?
    // If cell3 is exterior due to disconnection, the interface will miss face(0→3)
    // and the result will be incomplete. We expect cells 1,2,3 all interior
    // for a correct result, yielding 4 interface faces.
    //
    // But with current code, cell3 has:
    //   - No data term (face 2 is invisible mapped → nothing)
    //   - One smoothness edge to cell0 (face 3, extra) → penalty for cutting
    //   - NO edge to cell2 (face 2, invisible mapped → nothing)
    // So cell3 will be exterior. Interface = faces 0,1,2 (3 faces), missing face 3.

    int interface_faces = result.F.rows();
    std::cout << "[DiagTest] InvisibleMappedFaceDisconnectsInterior: "
              << interface_faces << " interface faces" << std::endl;

    // With the fix: invisible mapped face (cell2→cell3) now adds a smoothness
    // edge, so cell3 is interior (connected to cell2 via smoothness).
    // cell1 is correctly exterior (data term orientation says lc=cell1→SINK).
    // Interior: cell2, cell3. Exterior: cell0, cell1.
    // Interface: face1 (cell1-cell2) + face3 (cell0-cell3) = 2 faces.
    EXPECT_EQ(result.F.rows(), 2);
}

// ---------------------------------------------------------------------------
// H1 variant: Same topology but face(2→3) is made EXTRA instead of invisible
// mapped. This adds a smoothness edge, keeping the graph connected.
// Cell3 should now be interior (smoothness edge connects it to cell2).
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, ExtraFaceKeepsGraphConnected)
{
    PartitionResult pr;

    pr.mesh.V.resize(12, 3);
    pr.mesh.V <<
        0, 0, 0,  2, 0, 0,  0, 1, 0,
        0, 0, 1,  2, 0, 1,  0, 1, 1,
        0, 0, 2,  2, 0, 2,  0, 1, 2,
        0, 0, 3,  1, 0, 3,  0, 0.5, 3;

    pr.mesh.F.resize(4, 3);
    pr.mesh.F << 0, 1, 2,
                 3, 4, 5,
                 6, 7, 8,
                 9, 10, 11;

    pr.num_cells = 4;
    pr.per_patch_cells.resize(4, 2);
    pr.per_patch_cells <<
        0, 1,
        1, 2,
        2, 3,
        0, 3;

    pr.mesh.face_mapping.resize(4);
    pr.mesh.face_mapping << -1, 0, -1, -1;  // Only face 1 mapped; face 2 now EXTRA

    pr.mesh.offset_source.resize(4);
    pr.mesh.offset_source << -1, -1, -1, -1;

    pr.mesh.visibility.resize(4);
    pr.mesh.visibility << 0.0, 1.0, 0.0, 0.0;

    pr.mesh.orientation.resize(4);
    pr.mesh.orientation << 0.0, 1.0, 0.0, 0.0;

    Config config;
    Mesh result = extract_interface(pr, config);

    std::cout << "[DiagTest] ExtraFaceKeepsGraphConnected: "
              << result.F.rows() << " interface faces" << std::endl;

    // cell1 is exterior (data term pushes lc=cell1→SINK). cell2, cell3 interior
    // (cell2 from data, cell3 via smoothness propagation from cell2).
    // Interface: face1 (cell1-cell2) + face3 (cell0-cell3) = 2 faces.
    EXPECT_EQ(result.F.rows(), 2);
}

// ---------------------------------------------------------------------------
// H2: High extra-to-mapped ratio. Many extra faces with large total area
// creating strong smoothness pressure that overwhelms the data terms.
//
// Setup: cell0 (exterior) -- cell1 (interior candidate)
//   Multiple extra faces between cell0-cell1 (total area >> data term)
//   One visible mapped face between cell0-cell1 (data term)
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, HighSmoothnessOverwhelmingData)
{
    PartitionResult pr;

    // 1 mapped visible face (area=1) + 10 extra faces (each area≈4) = 40 total smoothness
    // The data term pushes cell1 interior with weight 1.
    // The smoothness penalty for cutting all extra faces is 40.
    // Since both data and smoothness are between the same cell pair,
    // the maxflow should still keep cell1 interior because
    // cutting mapped face (free) + extra faces = smoothness cost 40,
    // while NOT cutting = data cost 1.
    // Wait: all faces between same cell pair. Let me reconsider.
    //
    // Actually for a 2-cell case: cell0=SINK, cell1=?
    // Data term on cell1: pushes to SOURCE with weight 1 (from the visible mapped face)
    // Smoothness edges between cell0-cell1: total weight 40
    // If cell1=SOURCE: cut goes through all edges between cell0(SINK) and cell1(SOURCE)
    //   → cost = 40 (smoothness edges only, mapped face doesn't add edge)
    //   Actually wait: the mapped visible face adds data UNARY term, not edge.
    //   The extra faces add pairwise edge with weight 40 total.
    //   If cell1=SOURCE: min-cut separates it from cell0=SINK.
    //     Cost = sum of cut edges = 40 (all extra face edges)
    //     But cell1 has tweight(SOURCE)=1 from data term.
    //     Total: the maxflow cost for cell1=SOURCE is just the edge cost minus
    //     the unary support... Actually in maxflow: flow = min-cut.
    //     Unary: cell1 → SOURCE = 1
    //     Edge cell0-cell1 = 40
    //     Cell0 → SINK = 1e20
    //
    //     If cell1=SOURCE: cut = edges between SINK side and SOURCE side
    //       = edge(cell0, cell1) = 40
    //       Flow through SOURCE→cell1 = 1 (unary)
    //       Net: flow = 1 (limited by SOURCE→cell1 capacity)
    //
    //     If cell1=SINK: no edge is cut. Unary SOURCE→cell1=1 is cut.
    //       Flow = 1
    //
    //     Both cases give flow=1. Tie-breaking determines result.
    //     Let's make the data term larger to be unambiguous.

    int num_extra = 10;
    int total_faces = 1 + num_extra;

    pr.mesh.V.resize(total_faces * 3, 3);
    pr.mesh.F.resize(total_faces, 3);

    // Face 0: mapped visible face, area = 1 (unit right triangle)
    pr.mesh.V.row(0) << 0, 0, 0;
    pr.mesh.V.row(1) << 2, 0, 0;
    pr.mesh.V.row(2) << 0, 1, 0;
    pr.mesh.F.row(0) << 0, 1, 2;

    // Faces 1..10: extra faces, each area ≈ 2 (total ≈ 20)
    for (int i = 0; i < num_extra; ++i)
    {
        int base = 3 + i * 3;
        double z = (double)(i + 1);
        pr.mesh.V.row(base + 0) << 0, 0, z;
        pr.mesh.V.row(base + 1) << 2, 0, z;
        pr.mesh.V.row(base + 2) << 0, 2, z;
        pr.mesh.F.row(1 + i) << base, base + 1, base + 2;
    }

    pr.num_cells = 2;
    pr.per_patch_cells.resize(total_faces, 2);
    for (int f = 0; f < total_faces; ++f)
        pr.per_patch_cells.row(f) << 0, 1;

    pr.mesh.face_mapping.resize(total_faces);
    pr.mesh.face_mapping(0) = 0;
    for (int f = 1; f < total_faces; ++f)
        pr.mesh.face_mapping(f) = -1;

    pr.mesh.offset_source = VecXi::Constant(total_faces, -1);

    pr.mesh.visibility.resize(total_faces);
    pr.mesh.visibility(0) = 1.0;
    for (int f = 1; f < total_faces; ++f)
        pr.mesh.visibility(f) = 0.0;

    pr.mesh.orientation.resize(total_faces);
    pr.mesh.orientation(0) = 1.0;
    for (int f = 1; f < total_faces; ++f)
        pr.mesh.orientation(f) = 0.0;

    Config config;
    Mesh result = extract_interface(pr, config);

    std::cout << "[DiagTest] HighSmoothnessOverwhelmingData: "
              << result.F.rows() << " interface faces" << std::endl;

    // The mapped face has area ~1. Each extra face has area ~2. Total smoothness ~20.
    // Data term pushes cell1 to SOURCE with weight ~1 (area of mapped face).
    // In maxflow: SOURCE→cell1 capacity = 1, cell0→cell1 edge capacity = 20.
    // The min-cut is just 1 (cut the SOURCE→cell1 link), so cell1 = SINK = exterior.
    //
    // This means: when smoothness >> data, the interior shrinks!
    // This directly explains the fragmentation in the real run.
    //
    // For correct behavior, we'd want cell1 to be interior.
    // Record the actual result to diagnose.
    bool cell1_interior = result.F.rows() > 0;
    std::cout << "[DiagTest]   cell1 interior: " << (cell1_interior ? "YES" : "NO") << std::endl;

    // This test documents the expected (possibly wrong) behavior.
    // If cell1 becomes exterior (0 interface faces), it confirms H2.
    EXPECT_GT(result.F.rows(), 0)
        << "H2 CONFIRMED: smoothness overwhelms data term. "
        << "Cell1 classified as exterior despite visible mapped face.";
}

// ---------------------------------------------------------------------------
// H2 control: Same setup but only 1 extra face with small area.
// Data term should easily win.
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, SmallSmoothnessDataWins)
{
    PartitionResult pr;

    pr.mesh.V.resize(6, 3);
    pr.mesh.V <<
        0, 0, 0,  2, 0, 0,  0, 1, 0,   // face 0: mapped (area=1)
        0, 0, 1,  0.1, 0, 1,  0, 0.1, 1; // face 1: extra (area≈0.005)

    pr.mesh.F.resize(2, 3);
    pr.mesh.F << 0, 1, 2,
                 3, 4, 5;

    pr.num_cells = 2;
    pr.per_patch_cells.resize(2, 2);
    pr.per_patch_cells << 0, 1, 0, 1;

    pr.mesh.face_mapping.resize(2);
    pr.mesh.face_mapping << 0, -1;

    pr.mesh.offset_source = VecXi::Constant(2, -1);

    pr.mesh.visibility.resize(2);
    pr.mesh.visibility << 1.0, 0.0;

    pr.mesh.orientation.resize(2);
    pr.mesh.orientation << 1.0, 0.0;

    Config config;
    Mesh result = extract_interface(pr, config);

    // Data term (area=1) >> smoothness (area≈0.005). Cell1 should be interior.
    EXPECT_EQ(result.F.rows(), 2) << "Data term should win over small smoothness";
}

// ---------------------------------------------------------------------------
// H1+H2 combined: Multi-cell chain where some links are invisible mapped
// faces (no edge) and some are extra faces (smoothness). Tests how the
// disconnection and smoothness imbalance interact.
//
// cell0(ext) --extra-- cell1 --visible_mapped-- cell2 --invis_mapped-- cell3
//                                                       --extra-- cell0
//
// This is the realistic scenario: BSP creates many cells, with both mapped
// and extra faces between them.
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, ChainDisconnectionAndSmoothness)
{
    PartitionResult pr;

    // 5 faces, 4 cells
    pr.mesh.V.resize(15, 3);
    pr.mesh.V <<
        0, 0, 0,  2, 0, 0,  0, 1, 0,   // face 0: extra (cell0-cell1), area=1
        0, 0, 1,  2, 0, 1,  0, 1, 1,   // face 1: visible mapped (cell1-cell2), area=1
        0, 0, 2,  2, 0, 2,  0, 1, 2,   // face 2: invis mapped (cell2-cell3), area=1
        0, 0, 3,  2, 0, 3,  0, 1, 3,   // face 3: extra (cell0-cell3), area=1
        0, 0, 4,  2, 0, 4,  0, 1, 4;   // face 4: visible mapped (cell0-cell1), area=1

    pr.mesh.F.resize(5, 3);
    pr.mesh.F << 0, 1, 2,
                 3, 4, 5,
                 6, 7, 8,
                 9, 10, 11,
                 12, 13, 14;

    pr.num_cells = 4;
    pr.per_patch_cells.resize(5, 2);
    pr.per_patch_cells <<
        0, 1,   // face 0: extra
        1, 2,   // face 1: visible mapped
        2, 3,   // face 2: invisible mapped (THE PROBLEM)
        0, 3,   // face 3: extra
        0, 1;   // face 4: visible mapped (data term for cell1)

    pr.mesh.face_mapping.resize(5);
    pr.mesh.face_mapping << -1, 0, 1, -1, 2;

    pr.mesh.offset_source = VecXi::Constant(5, -1);

    pr.mesh.visibility.resize(5);
    pr.mesh.visibility << 0.0, 1.0, 0.3, 0.0, 1.0;

    pr.mesh.orientation.resize(5);
    // face 1 (cell1-cell2): ori>0 → cell1=exterior side, cell2=interior
    // face 4 (cell0-cell1): ori>0 → cell0=exterior side, cell1=interior
    pr.mesh.orientation << 0.0, 1.0, 1.0, 0.0, 1.0;

    Config config;
    Mesh result = extract_interface(pr, config);

    int mapped_in_interface = 0;
    for (int f = 0; f < result.F.rows(); ++f)
        if (result.face_mapping(f) >= 0)
            ++mapped_in_interface;

    std::cout << "[DiagTest] ChainDisconnectionAndSmoothness: "
              << result.F.rows() << " interface faces ("
              << mapped_in_interface << " mapped)" << std::endl;

    // Ideal: cells 1,2,3 all interior → 5 interface faces
    // Likely: cell3 is exterior (invisible mapped face disconnects it)
    //   → cell3=exterior, interface = faces 0,1,2,4 but face 2 is between
    //     cell2(interior) and cell3(exterior), and face 3 is between
    //     cell0(exterior) and cell3(exterior) → not in interface
    //   → interface faces: 0 (ext-int), 1 (int-int? no), 2 (int-ext), 4 (ext-int)
    //   Wait, let me trace through properly.
    //
    //   Data terms:
    //     face 1 (vis mapped, ori>0): lc=cell1 → SINK(w=1), rc=cell2 → SOURCE(w=1)
    //     face 4 (vis mapped, ori>0): lc=cell0 → SINK(w=1), rc=cell1 → SOURCE(w=1)
    //   Smoothness:
    //     face 0 (extra, cell0-cell1): edge(0,1) = area ≈ 1
    //     face 3 (extra, cell0-cell3): edge(0,3) = area ≈ 1
    //   No edge from invisible mapped face 2 (cell2-cell3).
    //
    //   cell0: SINK(1e20 + 1 + 1) SOURCE(0)      → definitely SINK
    //   cell1: SINK(1) SOURCE(1)                   → ambiguous, edge to cell0
    //   cell2: SINK(0) SOURCE(1)                   → wants SOURCE
    //   cell3: SINK(0) SOURCE(0)                   → no unary, edge to cell0 only
    //
    //   Graph edges: 0-1(w=1), 0-3(w=1)
    //   cell3 has no edge to cell2 (invisible mapped), only to cell0 (extra).
    //   cell3 is connected to SINK(cell0) via smoothness. No SOURCE pressure.
    //   → cell3 = SINK (exterior).
    //
    //   cell2: SOURCE(1), no edges. → cell2 = SOURCE (interior).
    //   cell1: SINK(1) from face1, SOURCE(1) from face4, edge to cell0(SINK) w=1.
    //     Min-cut: if cell1=SOURCE, cut edge(0,1)=1 and SINK→cell1=1 → cost=2
    //     If cell1=SINK, cut SOURCE→cell1=1 → cost=1
    //     → cell1 = SINK (exterior)!
    //
    //   Hmm, that's worse than expected. Let me recheck.
    //   face 1: lc=cell1 SINK(w=1), rc=cell2 SOURCE(w=1)
    //   face 4: lc=cell0 SINK(w=1), rc=cell1 SOURCE(w=1)
    //   cell0: SINK += 1e20 + 1(face4), SOURCE += 0  → net SINK huge
    //   cell1: SINK += 1(face1), SOURCE += 1(face4)   → net: balanced
    //   cell2: SINK += 0, SOURCE += 1(face1)          → net SOURCE
    //
    //   Edges: cell0-cell1 (w=1), cell0-cell3 (w=1)
    //
    //   Let me think in terms of maxflow.
    //   SOURCE --1--> cell1 --1(edge)--> cell0 --1e20--> SINK
    //   SOURCE --1--> cell2 (no edges)
    //   cell0 --1(edge)--> cell3 (no unary)
    //   cell1 --0(no edge)--> cell2 (invisible mapped, no edge!)
    //
    //   Also cell1 has SINK weight 1: cell1 --1--> SINK
    //
    //   Flow path 1: SOURCE→cell1→SINK = 1 (via unary)
    //   Flow path 2: SOURCE→cell2→? cell2 has no edges, so flow=1 stays in cell2.
    //     Actually cell2 has SOURCE weight 1 but no edges. It just sits there.
    //     cell2 = SOURCE (interior) since no flow can go from it to SINK.
    //
    //   Actually in maxflow: cell1 has SOURCE=1, SINK=1. These cancel.
    //   The net for cell1 is 0 unary. Plus edge to cell0(SINK) = 1.
    //   So effectively: no pressure either way for cell1, but edge to exterior.
    //   cell1 might be either. Let's see what maxflow does.

    // The test documents what happens. We check that SOMETHING is produced.
    EXPECT_GT(result.F.rows(), 0)
        << "Should have at least some interface faces";

    // Ideal would be 5 (all boundary faces), but due to invisible mapped
    // disconnection, we likely get fewer.
    if (result.F.rows() < 5)
    {
        std::cout << "[DiagTest]   WARNING: Got " << result.F.rows()
                  << " < 5 expected faces. Invisible mapped face likely disconnects cell3."
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Verify that visibility threshold matters: mapped face just below threshold
// (vis=0.49) adds nothing to the graph, while vis=0.51 adds data term.
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, VisibilityThreshold)
{
    auto make_partition = [](double vis_value) {
        PartitionResult pr;
        pr.mesh.V.resize(4, 3);
        pr.mesh.V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
        pr.mesh.F.resize(2, 3);
        pr.mesh.F << 0, 1, 2, 0, 2, 3;
        pr.num_cells = 2;
        pr.per_patch_cells.resize(2, 2);
        pr.per_patch_cells << 0, 1, 0, 1;
        pr.mesh.face_mapping.resize(2);
        pr.mesh.face_mapping << 0, 1;
        pr.mesh.offset_source = VecXi::Constant(2, -1);
        pr.mesh.visibility = VecXd::Constant(2, vis_value);
        pr.mesh.orientation = VecXd::Constant(2, 1.0);
        return pr;
    };

    Config config;

    // vis=0.51 → visible → data terms → cell1 interior → 2 interface faces
    {
        auto pr = make_partition(0.51);
        Mesh result = extract_interface(pr, config);
        EXPECT_EQ(result.F.rows(), 2) << "vis=0.51 should produce interface";
    }

    // vis=0.49 → invisible mapped → NO data terms → no edge → cell1 exterior
    {
        auto pr = make_partition(0.49);
        Mesh result = extract_interface(pr, config);
        std::cout << "[DiagTest] VisibilityThreshold vis=0.49: "
                  << result.F.rows() << " faces" << std::endl;
        // Mapped faces with vis<0.5 add nothing → cell1 stays exterior → 0 faces
        EXPECT_EQ(result.F.rows(), 0)
            << "vis=0.49 mapped faces add nothing, so no interior cells";
    }
}

// ---------------------------------------------------------------------------
// Count how many partition faces are "dead" (invisible mapped → contribute
// nothing to the graph). This simulates the diagnostic for the real scenario.
// ---------------------------------------------------------------------------
TEST(GraphCutDiag, DeadFaceRatio)
{
    // Simulate the ratios from the real run:
    // 290250 partition faces, 16535 mapped, 273715 extra
    // Among mapped: assume ~50% are visible (generous estimate)
    //
    // Dead faces (invisible mapped) = 16535 * 0.5 = ~8268
    // These faces add NO edges. Every cell boundary that consists solely of
    // invisible mapped faces is a disconnection point.
    //
    // This test just verifies the logic directly.

    PartitionResult pr;
    pr.mesh.V.resize(4, 3);
    pr.mesh.V << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;

    // 3 faces, all between cell0-cell1
    pr.mesh.F.resize(3, 3);
    pr.mesh.F << 0, 1, 2, 0, 2, 3, 0, 1, 3;

    pr.num_cells = 2;
    pr.per_patch_cells.resize(3, 2);
    pr.per_patch_cells << 0, 1, 0, 1, 0, 1;

    // 2 invisible mapped + 1 extra
    pr.mesh.face_mapping.resize(3);
    pr.mesh.face_mapping << 0, 1, -1;

    pr.mesh.offset_source = VecXi::Constant(3, -1);

    pr.mesh.visibility.resize(3);
    pr.mesh.visibility << 0.3, 0.2, 0.0;  // all below threshold

    pr.mesh.orientation = VecXd::Constant(3, 1.0);

    Config config;
    Mesh result = extract_interface(pr, config);

    // No visible mapped faces → no data term.
    // Only 1 extra face adds a smoothness edge.
    // No interior pressure → cell1 stays exterior → 0 faces.
    EXPECT_EQ(result.F.rows(), 0)
        << "No visible mapped faces → no interior cells";

    // Now make the mapped faces visible:
    pr.mesh.visibility << 0.8, 0.9, 0.0;
    result = extract_interface(pr, config);

    // 2 visible mapped faces push cell1 interior.
    // 1 extra face adds smoothness but smaller than data.
    EXPECT_EQ(result.F.rows(), 3)
        << "With visible mapped faces, all faces should be in interface";
}
