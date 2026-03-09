#include "graph_cut.h"
#include <igl/remove_unreferenced.h>
#include <maxflow.h>
#include <iostream>
#include <vector>

namespace vpmr
{

Mesh extract_interface(const PartitionResult& partition, const Config& config)
{
    Mesh mesh = partition.mesh;
    int nf = mesh.F.rows();
    int num_cells = partition.num_cells;
    // ppc(f, 0) = positive cell (normal side), ppc(f, 1) = negative cell
    const MatXi& ppc = partition.per_patch_cells;

    // Diagnostics
    int mapped_count = 0, extra_count = 0;
    for (int f = 0; f < nf; ++f)
    {
        if (mesh.face_mapping(f) >= 0)
            ++mapped_count;
        else
            ++extra_count;
    }
    std::cout << "[Graph Cut]   Classification: mapped=" << mapped_count
              << " extra=" << extra_count << std::endl;

    // Build graph: cells are nodes, cell 0 = unbounded = SINK
    maxflow::Graph_DDD graph(num_cells, nf);

    for (int c = 0; c < num_cells; ++c)
    {
        graph.add_node();
    }

    // Cell 0 is unbounded -> forced to SINK (exterior)
    graph.add_tweights(0, 0, 1e20);

    // First pass: compute total data and effective smoothness areas.
    //
    // All faces contribute smoothness (pairwise) to keep the graph fully
    // connected.  Smoothness is weighted by (1 - vis) so that visible faces
    // have near-zero smoothness cost (the cut naturally goes through them)
    // while extra/invisible faces have high smoothness cost (the cut avoids
    // them, keeping the interior coherent).
    //
    // Only visible mapped faces additionally contribute data (unary).
    double total_data_area = 0;
    double total_smooth_area = 0; // effective: (1-vis)-weighted
    for (int f = 0; f < nf; ++f)
    {
        int lc = ppc(f, 0);
        int rc = ppc(f, 1);
        if (lc < 0 || rc < 0 || lc >= num_cells || rc >= num_cells)
            continue;
        if (lc == rc)
            continue;

        double area = mesh.face_area(f);
        if (area < 1e-15)
            continue;
        const bool is_mapped = mesh.face_mapping(f) >= 0;
        const double vis = (is_mapped && mesh.visibility.size() > 0) ? mesh.visibility(f) : 0.0;
        const bool is_visible = is_mapped && vis > 0.5;

        if (is_visible)
            total_data_area += area;
        // Smoothness weight: (1-vis) for mapped, 1 for extra
        double smooth_w = is_mapped ? (1.0 - vis) : 1.0;
        total_smooth_area += smooth_w * area;
    }

    // Lambda balances data vs effective smoothness.  The 0.8 factor gives
    // the data term a slight edge so that ties break toward interior,
    // preventing cells with visible faces from defaulting to exterior.
    double lambda = 1.0;
    if (total_smooth_area > 0 && total_data_area > 0)
        lambda = 0.8 * total_data_area / total_smooth_area;

    std::cout << "[Graph Cut]   Data area=" << total_data_area
              << " smooth area=" << total_smooth_area
              << " lambda=" << lambda << std::endl;

    // Second pass: build the graph.
    for (int f = 0; f < nf; ++f)
    {
        int lc = ppc(f, 0);
        int rc = ppc(f, 1);

        if (lc < 0 || rc < 0 || lc >= num_cells || rc >= num_cells)
            continue;
        if (lc == rc)
            continue; // same cell on both sides — no cut possible

        double area = mesh.face_area(f);

        const bool is_mapped = mesh.face_mapping(f) >= 0;
        const double vis = (is_mapped && mesh.visibility.size() > 0) ? mesh.visibility(f) : 0.0;
        const bool is_visible = is_mapped && vis > 0.5;

        // Eq. 7: only visible mapped faces contribute unary terms.
        // Orientation determines which side of the face is interior/exterior.
        // orientation > 0: normal points outward → lc is exterior, rc is interior
        // orientation < 0: normal points inward  → lc is interior, rc is exterior
        if (is_visible)
        {
            double w_data = area;
            if (mesh.orientation(f) >= 0)
            {
                graph.add_tweights(lc, 0, w_data); // lc → SINK (exterior)
                graph.add_tweights(rc, w_data, 0);  // rc → SOURCE (interior)
            }
            else
            {
                graph.add_tweights(lc, w_data, 0); // lc → SOURCE (interior)
                graph.add_tweights(rc, 0, w_data);  // rc → SINK (exterior)
            }
        }

        // Eq. 8: ALL faces contribute pairwise smoothness, weighted by
        // (1-vis) so visible boundaries are cheap to cut (the data term
        // handles them) while extra/invisible boundaries are expensive
        // (keeping interior regions coherent).
        if (area > 1e-15)
        {
            double smooth_w = is_mapped ? (1.0 - vis) : 1.0;
            double w_smooth = lambda * smooth_w * area;
            if (w_smooth > 1e-15)
                graph.add_edge(lc, rc, w_smooth, w_smooth);
        }
    }

    graph.maxflow();

    // Diagnostic: count interior vs exterior cells
    {
        int n_interior = 0, n_exterior = 0;
        for (int c = 0; c < num_cells; ++c)
        {
            if (graph.what_segment(c) == maxflow::Graph_DDD::SOURCE)
                ++n_interior;
            else
                ++n_exterior;
        }
        std::cout << "[Graph Cut]   Cells: " << n_interior << " interior, " << n_exterior << " exterior (total " << num_cells << ")"
                  << std::endl;
    }

    // Extract interface: faces where adjacent cells have different labels
    std::vector<int> interface_faces;
    for (int f = 0; f < nf; ++f)
    {
        int lc = ppc(f, 0);
        int rc = ppc(f, 1);
        if (lc < 0 || rc < 0 || lc >= num_cells || rc >= num_cells)
            continue;

        bool lc_interior = (graph.what_segment(lc) == maxflow::Graph_DDD::SOURCE);
        bool rc_interior = (graph.what_segment(rc) == maxflow::Graph_DDD::SOURCE);

        if (lc_interior != rc_interior)
        {
            interface_faces.push_back(f);
        }
    }

    int iface_mapped = 0, iface_extra = 0;
    for (int f : interface_faces)
    {
        if (mesh.face_mapping(f) >= 0)
            ++iface_mapped;
        else
            ++iface_extra;
    }
    std::cout << "[Graph Cut]   Interface faces: " << interface_faces.size() << " (mapped=" << iface_mapped << " extra=" << iface_extra
              << ")" << std::endl;

    Mesh result;
    result.F.resize(interface_faces.size(), 3);
    result.face_mapping.resize(interface_faces.size());
    result.offset_source.resize(interface_faces.size());

    for (int i = 0; i < (int)interface_faces.size(); ++i)
    {
        int f = interface_faces[i];
        int lc = ppc(f, 0);
        bool lc_interior = (graph.what_segment(lc) == maxflow::Graph_DDD::SOURCE);

        // Orient face so normal points outward (from interior to exterior)
        // ppc(f,0) = positive cell = cell on normal side
        // If normal side is interior, normal points inward → flip for outward
        if (lc_interior)
        {
            result.F(i, 0) = mesh.F(f, 0);
            result.F(i, 1) = mesh.F(f, 2);
            result.F(i, 2) = mesh.F(f, 1);
        }
        else
        {
            result.F.row(i) = mesh.F.row(f);
        }

        result.face_mapping(i) = mesh.face_mapping(f);
        result.offset_source(i) = mesh.offset_source(f);
    }

    result.V = mesh.V;

    // Remove unreferenced vertices (partition mesh has many vertices not used by interface)
    {
        MatXd NV;
        MatXi NF;
        VecXi I;
        igl::remove_unreferenced(result.V, result.F, NV, NF, I);
        result.V = NV;
        result.F = NF;
    }

    return result;
}

} // namespace vpmr
