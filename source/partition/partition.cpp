#include "partition.h"
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/copyleft/cgal/extract_cells.h>
#include <igl/remove_unreferenced.h>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace vpmr
{

using Edge = std::pair<int, int>;

static Edge make_edge(int a, int b)
{
    return a < b ? Edge{a, b} : Edge{b, a};
}

PartitionResult partition_space(const Mesh& mesh, int original_face_count)
{
    // ========================================
    // Step 1: Resolve self-intersections
    // ========================================
    Eigen::MatrixXd VV;
    Eigen::MatrixXi FF, IF;
    Eigen::VectorXi J, IM;

    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.detect_only = false;
    params.first_only = false;

    igl::copyleft::cgal::remesh_self_intersections(mesh.V, mesh.F, params, VV, FF, IF, J, IM);

    // Remap vertices (merge coincident)
    for (int i = 0; i < FF.rows(); ++i)
        for (int j = 0; j < 3; ++j)
            FF(i, j) = IM(FF(i, j));

    // Remove unreferenced vertices
    Eigen::MatrixXd VV_clean;
    Eigen::MatrixXi FF_clean;
    Eigen::VectorXi I_map;
    igl::remove_unreferenced(VV, FF, VV_clean, FF_clean, I_map);

    // Remove degenerate faces (where 2+ vertices coincide after IM remap)
    {
        std::vector<int> good;
        for (int i = 0; i < FF_clean.rows(); ++i)
        {
            if (FF_clean(i, 0) != FF_clean(i, 1) && FF_clean(i, 1) != FF_clean(i, 2) && FF_clean(i, 0) != FF_clean(i, 2))
                good.push_back(i);
        }
        if ((int)good.size() < FF_clean.rows())
        {
            Eigen::MatrixXi FF_good(good.size(), 3);
            Eigen::VectorXi J_good(good.size());
            for (int i = 0; i < (int)good.size(); ++i)
            {
                FF_good.row(i) = FF_clean.row(good[i]);
                J_good(i) = J(good[i]);
            }
            FF_clean = FF_good;
            J = J_good;
        }
    }

    int nf_clean = FF_clean.rows();

    std::cout << "[Partition]   After remesh_self_intersections: " << VV_clean.rows() << " vertices, " << nf_clean << " faces" << std::endl;

    // ========================================
    // Step 2: Extract cells using libigl
    // ========================================
    Eigen::MatrixXi cells;
    int num_cells = (int)igl::copyleft::cgal::extract_cells(VV_clean, FF_clean, cells);

    std::cout << "[Partition]   extract_cells: " << num_cells << " cells" << std::endl;

    // ========================================
    // Build PartitionResult
    // ========================================
    PartitionResult result;
    result.mesh.V = VV_clean;
    result.mesh.F = FF_clean;
    result.per_patch_cells = cells;  // cells(f,0) = positive cell, cells(f,1) = negative cell
    result.num_cells = num_cells;

    // ========================================
    // Build face_mapping: partition face → original input face
    // ========================================
    // J maps remeshed face → original mesh face. After remesh_self_intersections,
    // each output face comes from splitting an input face. J(i) gives the
    // original face index.

    int nf_out = FF_clean.rows();
    result.mesh.face_mapping.resize(nf_out);

    int mapped_count = 0;
    for (int i = 0; i < nf_out; ++i)
    {
        if (i < (int)J.size())
        {
            int orig = J(i);
            if (orig >= 0 && orig < original_face_count)
            {
                result.mesh.face_mapping(i) = orig;
                ++mapped_count;
            }
            else
            {
                result.mesh.face_mapping(i) = -1;
            }
        }
        else
        {
            result.mesh.face_mapping(i) = -1;
        }
    }

    int extra_count = nf_out - mapped_count;
    std::cout << "[Partition]   face_mapping: " << mapped_count << " mapped, " << extra_count << " extra" << std::endl;

    // Propagate offset_source using the same face_mapping
    result.mesh.offset_source.resize(nf_out);
    for (int i = 0; i < nf_out; ++i)
    {
        int orig = result.mesh.face_mapping(i);
        if (orig >= 0 && orig < mesh.offset_source.size() && mesh.offset_source(orig) >= 0)
        {
            result.mesh.offset_source(i) = mesh.offset_source(orig);
        }
        else
        {
            result.mesh.offset_source(i) = -1;
        }
    }

    return result;
}

} // namespace vpmr
