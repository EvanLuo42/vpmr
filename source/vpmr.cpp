#include "vpmr/vpmr.h"
#include "visual_measures/visual_measures.h"
#include "orientation/orientation.h"
#include "offset/offset.h"
#include "partition/partition.h"
#include "graph_cut/graph_cut.h"
#include "simplification/simplification.h"
#include "topology/topology.h"
#include "attributes/attributes.h"
#include <igl/remove_unreferenced.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/boundary_loop.h>
#include <iostream>

namespace vpmr
{

Mesh run_pipeline(Mesh input, const Config& config)
{
    Mesh mesh = std::move(input);

    // Weld coincident vertices (e.g. STL triangle soup has 3 unique verts per face)
    {
        MatXd SV;
        MatXi SF;
        VecXi SVI, SVJ;
        igl::remove_duplicate_vertices(mesh.V, mesh.F, 0.0, SV, SVI, SVJ, SF);
        if (SV.rows() < mesh.V.rows())
        {
            std::cout << "[Preprocess] Welded duplicate vertices: " << mesh.V.rows() << " -> " << SV.rows() << std::endl;
            mesh.V = SV;
            mesh.F = SF;
        }
    }

    // §3.1: Compute visual measures
    std::cout << "[Visual Measures] Computing visual measures..." << std::endl;
    compute_visual_measures(mesh, config);
    std::cout << "[Visual Measures]   Faces: " << mesh.F.rows() << std::endl;

    // §3.2: Orientation adjustment
    std::cout << "[Orientation Adjustment] Adjusting orientation..." << std::endl;
    adjust_orientation(mesh);

    // §3.3: Offset open surfaces
    std::cout << "[Offset] Offsetting open surfaces..." << std::endl;
    Mesh input_copy = mesh; // keep copy for attribute recovery
    int original_face_count = mesh.F.rows();
    offset_open_surfaces(mesh, config);
    std::cout << "[Offset]   Faces after offset: " << mesh.F.rows() << " (original: " << original_face_count << ")" << std::endl;

    // §3.4: Space partition (resolve self-intersections + extract cells)
    std::cout << "[Partition] Partitioning space..." << std::endl;
    auto partition = partition_space(mesh, original_face_count);
    std::cout << "[Partition]   Partition faces: " << partition.mesh.F.rows() << ", cells: " << partition.num_cells << std::endl;

    Mesh result;
    if (partition.num_cells >= 2)
    {
        // §3.5.1: Recompute visual measures on partition mesh
        std::cout << "[Visual Measures] Recomputing on partition mesh..." << std::endl;
        compute_visual_measures(partition.mesh, mesh, config);

        // §3.5.2: Re-orient patches on partition mesh
        std::cout << "[Orientation] Re-adjusting orientation on partition mesh..." << std::endl;
        adjust_orientation(partition.mesh, &partition.per_patch_cells);

        // §3.5.3: Graph cut + interface extraction
        std::cout << "[Graph Cut] Extracting interface mesh..." << std::endl;
        result = extract_interface(partition, config);
        std::cout << "[Graph Cut]   Interface faces: " << result.F.rows() << std::endl;

        // If graph cut produced empty result, fall back to pre-partition mesh
        if (result.F.rows() == 0)
        {
            std::cout << "[Graph Cut]   Graph cut produced empty result, using pre-partition mesh" << std::endl;
            result.V = mesh.V;
            result.F = mesh.F;

            result.face_mapping.resize(mesh.F.rows());
            for (int f = 0; f < mesh.F.rows(); ++f)
                result.face_mapping(f) = (f < original_face_count) ? f : -1;

            if (mesh.offset_source.size() == mesh.F.rows())
                result.offset_source = mesh.offset_source;
            else
            {
                result.offset_source.resize(mesh.F.rows());
                result.offset_source.setConstant(-1);
            }
        }
    }
    else
    {
        // Single cell = no interior. Use the oriented + hole-filled mesh directly.
        std::cout << "[Graph Cut]   Only 1 cell, skipping graph cut" << std::endl;
        result = mesh;
    }

    // §3.6: Constrained simplification
    std::cout << "[Simplification] Simplifying mesh..." << std::endl;
    simplify_mesh(result, config);
    std::cout << "[Simplification]   Simplified faces: " << result.F.rows() << std::endl;

    // Fill any remaining boundary holes before topology fixing
    {
        std::vector<std::vector<int>> loops;
        igl::boundary_loop(result.F, loops);
        if (!loops.empty())
        {
            int added = 0;
            for (auto& loop : loops)
                if ((int)loop.size() >= 3)
                    added += (int)loop.size() - 2;

            if (added > 0)
            {
                int orig_nf = result.F.rows();
                MatXi newF(orig_nf + added, 3);
                newF.topRows(orig_nf) = result.F;

                VecXi newFM(orig_nf + added);
                if (result.face_mapping.size() == orig_nf)
                    newFM.head(orig_nf) = result.face_mapping;
                else
                    newFM.head(orig_nf).setConstant(-1);

                VecXi newOS(orig_nf + added);
                if (result.offset_source.size() == orig_nf)
                    newOS.head(orig_nf) = result.offset_source;
                else
                    newOS.head(orig_nf).setConstant(-1);

                int fi = orig_nf;
                for (auto& loop : loops)
                {
                    for (int i = 1; i + 1 < (int)loop.size(); ++i)
                    {
                        newF.row(fi) << loop[0], loop[i + 1], loop[i];
                        newFM(fi) = -1;
                        newOS(fi) = -1;
                        ++fi;
                    }
                }
                result.F = newF;
                result.face_mapping = newFM;
                result.offset_source = newOS;
                std::cout << "[Simplification]   Filled " << loops.size() << " boundary loops (" << added << " faces)" << std::endl;
            }
        }
    }

    // §3.7: Topological correction
    std::cout << "[Topology Fix] Fixing topology..." << std::endl;
    fix_topology(result);
    std::cout << "[Topology Fix]   Faces after topology fix: " << result.F.rows() << std::endl;

    // §3.8: Attribute recovery
    std::cout << "[Attribute Recovery] Recovering attributes..." << std::endl;
    recover_attributes(result, input_copy);

    std::cout << "Done. Output: " << result.V.rows() << " vertices, " << result.F.rows() << " faces." << std::endl;

    return result;
}

} // namespace vpmr
