#include "partition.h"
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include <igl/copyleft/cgal/extract_cells.h>
#include <igl/remove_unreferenced.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>

namespace vpmr
{

PartitionResult partition_space(const Mesh& mesh, int original_face_count)
{
    int nf = mesh.F.rows();

    // ========================================
    // Step 1: Resolve self-intersections (§3.4)
    // ========================================
    Eigen::MatrixXd VV;
    Eigen::MatrixXi FF;
    Eigen::MatrixXi IF; // intersecting face pairs
    Eigen::VectorXi J;  // birth face: FF(i) comes from F(J(i))
    Eigen::VectorXi IM; // unique vertex mapping

    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.detect_only = false;
    params.stitch_all = true;

    std::cout << "[Partition]   Resolving self-intersections..." << std::endl;
    igl::copyleft::cgal::remesh_self_intersections(
        mesh.V, mesh.F, params, VV, FF, IF, J, IM);

    std::cout << "[Partition]   Self-intersections resolved: "
              << nf << " → " << FF.rows() << " faces, "
              << IF.rows() << " intersecting pairs" << std::endl;

    // Apply IM to merge duplicate vertices
    for (int i = 0; i < FF.rows(); ++i)
        for (int j = 0; j < 3; ++j)
            FF(i, j) = IM(FF(i, j));

    // ========================================
    // Step 2: Clean up mesh
    // ========================================
    Eigen::MatrixXd V_clean;
    Eigen::MatrixXi F_clean;
    Eigen::VectorXi I_map;
    igl::remove_unreferenced(VV, FF, V_clean, F_clean, I_map);

    // Remove degenerate faces (where 2+ vertices coincide after merging)
    {
        std::vector<int> good;
        for (int i = 0; i < F_clean.rows(); ++i)
        {
            if (F_clean(i, 0) != F_clean(i, 1) &&
                F_clean(i, 1) != F_clean(i, 2) &&
                F_clean(i, 0) != F_clean(i, 2))
                good.push_back(i);
        }
        if ((int)good.size() < F_clean.rows())
        {
            int removed = F_clean.rows() - (int)good.size();
            Eigen::MatrixXi FF_good(good.size(), 3);
            Eigen::VectorXi J_good(good.size());
            for (int i = 0; i < (int)good.size(); ++i)
            {
                FF_good.row(i) = F_clean.row(good[i]);
                J_good(i) = J(good[i]);
            }
            F_clean = FF_good;
            J = J_good;
            std::cout << "[Partition]   Removed " << removed << " degenerate faces" << std::endl;
        }
    }

    // ========================================
    // Step 3: Extract cells (§3.4)
    // ========================================
    std::cout << "[Partition]   Extracting cells..." << std::endl;
    Eigen::MatrixXi cells;
    size_t num_cells = igl::copyleft::cgal::extract_cells(V_clean, F_clean, cells);

    // ========================================
    // Build PartitionResult
    // ========================================
    PartitionResult result;
    result.mesh.V = V_clean;
    result.mesh.F = F_clean;
    result.per_patch_cells = cells;
    result.num_cells = (int)num_cells;

    int nf_out = F_clean.rows();

    // Build face_mapping from J (birth face vector)
    // J(i) = index of the input face that gave birth to output face i
    result.mesh.face_mapping.resize(nf_out);
    int mapped_count = 0;
    for (int i = 0; i < nf_out; ++i)
    {
        int birth = J(i);
        if (birth >= 0 && birth < original_face_count)
        {
            result.mesh.face_mapping(i) = birth;
            ++mapped_count;
        }
        else
        {
            result.mesh.face_mapping(i) = -1;
        }
    }

    // Propagate offset_source using J
    result.mesh.offset_source.resize(nf_out);
    for (int i = 0; i < nf_out; ++i)
    {
        int birth = J(i);
        if (birth >= 0 && birth < mesh.offset_source.size() && mesh.offset_source(birth) >= 0)
        {
            result.mesh.offset_source(i) = mesh.offset_source(birth);
        }
        else
        {
            result.mesh.offset_source(i) = -1;
        }
    }

    int extra_count = nf_out - mapped_count;
    std::cout << "[Partition]   Partition: " << V_clean.rows() << " vertices, "
              << nf_out << " faces, " << num_cells << " cells" << std::endl;
    std::cout << "[Partition]   face_mapping: " << mapped_count << " mapped, "
              << extra_count << " extra" << std::endl;

    return result;
}

} // namespace vpmr
