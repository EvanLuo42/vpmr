#include "graph_cut.h"
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

    // Classify faces:
    // VISIBLE: mapped to an input face and visible on M_partition
    // INVISIBLE: mapped to an input face and invisible on M_partition
    // EXTRA: no strict mapping M(f)
    enum FaceClass
    {
        VISIBLE,
        INVISIBLE,
        EXTRA
    };
    std::vector<FaceClass> face_class(nf);

    int vis_count = 0, invis_count = 0, extra_count = 0;
    for (int f = 0; f < nf; ++f)
    {
        if (mesh.face_mapping(f) >= 0)
        {
            if (mesh.visibility(f) > 0.5)
            {
                face_class[f] = VISIBLE;
                ++vis_count;
            }
            else
            {
                face_class[f] = INVISIBLE;
                ++invis_count;
            }
        }
        else
        {
            face_class[f] = EXTRA;
            ++extra_count;
        }
    }

    // Diagnostics
    int skipped_faces = 0;
    std::vector<int> lc_hist(num_cells, 0), rc_hist(num_cells, 0);
    int same_cell_count = 0;
    for (int f = 0; f < nf; ++f)
    {
        int lc = ppc(f, 0);
        int rc = ppc(f, 1);
        if (lc < 0 || rc < 0 || lc >= num_cells || rc >= num_cells)
        {
            ++skipped_faces;
            continue;
        }
        lc_hist[lc]++;
        rc_hist[rc]++;
        if (lc == rc) ++same_cell_count;
    }

    std::cout << "[Graph Cut]   Classification: visible=" << vis_count << " invisible=" << invis_count << " extra=" << extra_count
              << " skipped=" << skipped_faces << " same_cell=" << same_cell_count << std::endl;
    for (int c = 0; c < std::min(num_cells, 10); ++c)
    {
        std::cout << "[Graph Cut]   ppc hist cell " << c << ": lc=" << lc_hist[c] << " rc=" << rc_hist[c] << std::endl;
    }

    // Build graph: cells are nodes, cell 0 = unbounded = SINK
    maxflow::Graph_DDD graph(num_cells, nf);

    for (int c = 0; c < num_cells; ++c)
    {
        graph.add_node();
    }

    // Cell 0 is unbounded -> forced to SINK (exterior)
    graph.add_tweights(0, 0, 1e20);

    // Track per-cell source/sink weights for diagnostics
    std::vector<double> cell_source(num_cells, 0), cell_sink(num_cells, 0);
    cell_sink[0] += 1e20;

    for (int f = 0; f < nf; ++f)
    {
        int lc = ppc(f, 0);
        int rc = ppc(f, 1);

        if (lc < 0 || rc < 0 || lc >= num_cells || rc >= num_cells)
            continue;
        if (lc == rc)
            continue; // same cell on both sides — no cut possible

        double w = mesh.face_area(f);

        if (face_class[f] == VISIBLE)
        {
            // Data term (Eq. 7): use orientation to determine normal direction.
            // orientation > 0: normal points outward → lc is exterior, rc is interior
            // orientation < 0: normal points inward  → lc is interior, rc is exterior
            if (mesh.orientation(f) >= 0)
            {
                graph.add_tweights(lc, 0, w); // lc → SINK (exterior)
                graph.add_tweights(rc, w, 0); // rc → SOURCE (interior)
                cell_sink[lc] += w;
                cell_source[rc] += w;
            }
            else
            {
                graph.add_tweights(lc, w, 0); // lc → SOURCE (interior)
                graph.add_tweights(rc, 0, w); // rc → SINK (exterior)
                cell_source[lc] += w;
                cell_sink[rc] += w;
            }
        }
        else if (face_class[f] == EXTRA)
        {
            // Smoothness term (Eq. 8): only true extra faces get edges.
            graph.add_edge(lc, rc, w, w);
        }
        // INVISIBLE faces: no edge, no data term (§3.5.3)
    }

    // Debug: print per-cell weights
    for (int c = 0; c < std::min(num_cells, 10); ++c)
    {
        std::cout << "[Graph Cut]   Cell " << c << ": source=" << cell_source[c]
                  << " sink=" << cell_sink[c]
                  << " net=" << (cell_source[c] - cell_sink[c]) << std::endl;
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
        {
            ++iface_mapped;
        }
        else
        {
            ++iface_extra;
        }
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
    return result;
}

} // namespace vpmr
