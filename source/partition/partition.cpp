#include "partition.h"
#include "bsp.h"
#include <igl/remove_unreferenced.h>
#include <Eigen/Core>
#include <iostream>

namespace vpmr
{

PartitionResult partition_space(const Mesh& mesh, int original_face_count)
{
    int nf = mesh.F.rows();

    // ========================================
    // Step 1: Initialize BSP from Delaunay tetrahedralization (§3.4)
    // ========================================
    BSPTree bsp;
    bsp.init_delaunay(mesh.V);

    // ========================================
    // Step 2: Insert each mesh face into the BSP tree
    // ========================================
    for (int f = 0; f < nf; ++f)
    {
        bsp.insert_face(f, mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2));
        if ((f + 1) % 1000 == 0 || f + 1 == nf)
        {
            std::cout << "\r[Partition]   Inserted " << (f + 1) << " / " << nf << " faces" << std::flush;
        }
    }
    std::cout << std::endl;

    // ========================================
    // Step 3: Extract partition mesh with cell adjacency
    // ========================================
    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out, per_face_cells;
    Eigen::VectorXi face_labels;
    int num_cells;

    bsp.extract_partition(V_out, F_out, per_face_cells, face_labels, num_cells);

    // ========================================
    // Step 4: Remove unreferenced vertices
    // ========================================
    Eigen::MatrixXd V_clean;
    Eigen::MatrixXi F_clean;
    Eigen::VectorXi I_map;
    igl::remove_unreferenced(V_out, F_out, V_clean, F_clean, I_map);

    // Remove degenerate faces (where 2+ vertices coincide after double conversion)
    {
        std::vector<int> good;
        for (int i = 0; i < F_clean.rows(); ++i)
        {
            if (F_clean(i, 0) != F_clean(i, 1) && F_clean(i, 1) != F_clean(i, 2) && F_clean(i, 0) != F_clean(i, 2))
                good.push_back(i);
        }
        if ((int)good.size() < F_clean.rows())
        {
            int removed = F_clean.rows() - (int)good.size();
            Eigen::MatrixXi FF_good(good.size(), 3);
            Eigen::MatrixXi PFC_good(good.size(), 2);
            Eigen::VectorXi FL_good(good.size());
            for (int i = 0; i < (int)good.size(); ++i)
            {
                FF_good.row(i) = F_clean.row(good[i]);
                PFC_good.row(i) = per_face_cells.row(good[i]);
                FL_good(i) = face_labels(good[i]);
            }
            F_clean = FF_good;
            per_face_cells = PFC_good;
            face_labels = FL_good;
            std::cout << "[Partition]   Removed " << removed << " degenerate faces" << std::endl;
        }
    }

    // ========================================
    // Build PartitionResult
    // ========================================
    PartitionResult result;
    result.mesh.V = V_clean;
    result.mesh.F = F_clean;
    result.per_patch_cells = per_face_cells;
    result.num_cells = num_cells;

    int nf_out = F_clean.rows();

    // Build face_mapping from BSP face labels
    // label >= 0 && < original_face_count → maps to original input face
    // otherwise → extra face (offset/stitching/BSP artifact)
    result.mesh.face_mapping.resize(nf_out);

    int mapped_count = 0;
    for (int i = 0; i < nf_out; ++i)
    {
        int label = face_labels(i);
        if (label >= 0 && label < original_face_count)
        {
            result.mesh.face_mapping(i) = label;
            ++mapped_count;
        }
        else
        {
            result.mesh.face_mapping(i) = -1;
        }
    }

    int extra_count = nf_out - mapped_count;
    std::cout << "[Partition]   BSP partition: " << V_clean.rows() << " vertices, "
              << nf_out << " faces, " << num_cells << " cells" << std::endl;
    std::cout << "[Partition]   face_mapping: " << mapped_count << " mapped, "
              << extra_count << " extra" << std::endl;

    // Propagate offset_source using BSP face labels (not face_mapping,
    // since offset faces have labels >= original_face_count)
    result.mesh.offset_source.resize(nf_out);
    for (int i = 0; i < nf_out; ++i)
    {
        int label = face_labels(i);
        if (label >= 0 && label < mesh.offset_source.size() && mesh.offset_source(label) >= 0)
        {
            result.mesh.offset_source(i) = mesh.offset_source(label);
        }
        else
        {
            result.mesh.offset_source(i) = -1;
        }
    }

    return result;
}

} // namespace vpmr
