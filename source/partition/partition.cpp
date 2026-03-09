#include "partition.h"
#include "bsp.h"
#include <igl/remove_unreferenced.h>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

namespace vpmr
{

// Check if point p lies inside triangle (a, b, c) in 3D.
// Assumes p is already on (or very near) the plane of the triangle.
static bool point_in_triangle_3d(
    const Vec3d& p, const Vec3d& a, const Vec3d& b, const Vec3d& c, double tol)
{
    Vec3d ab = b - a, ac = c - a, ap = p - a;
    double d00 = ab.dot(ab);
    double d01 = ab.dot(ac);
    double d11 = ac.dot(ac);
    double d20 = ap.dot(ab);
    double d21 = ap.dot(ac);
    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-30)
        return false;
    double u = (d11 * d20 - d01 * d21) / denom;
    double v = (d00 * d21 - d01 * d20) / denom;
    return u >= -tol && v >= -tol && (u + v) <= 1.0 + tol;
}

// Check if point p lies on the supporting plane of triangle (a, b, c).
static bool point_on_plane(
    const Vec3d& p, const Vec3d& a, const Vec3d& b, const Vec3d& c, double tol)
{
    Vec3d n = (b - a).cross(c - a);
    double area2 = n.norm();
    if (area2 < 1e-30)
        return false;
    return std::abs(n.dot(p - a)) / area2 < tol;
}

PartitionResult partition_space(const Mesh& mesh, int original_face_count)
{
    int nv = mesh.V.rows();
    int nf = mesh.F.rows();

    // ========================================
    // Step 1: Build BSP tree (§3.4)
    // ========================================
    std::cout << "[Partition]   Initializing Delaunay tetrahedralization..." << std::endl;
    BSPTree bsp;
    bsp.init_delaunay(mesh.V);

    // Insert all input faces into BSP
    std::cout << "[Partition]   Inserting " << nf << " faces into BSP..." << std::endl;
    {
        auto t0 = std::chrono::steady_clock::now();
        int report_interval = std::max(1, nf / 20); // report ~20 times
        for (int i = 0; i < nf; ++i)
        {
            bsp.insert_face(i, mesh.F(i, 0), mesh.F(i, 1), mesh.F(i, 2));
            if ((i + 1) % report_interval == 0 || i + 1 == nf)
            {
                auto t1 = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t1 - t0).count();
                double rate = (i + 1) / elapsed;
                double eta = (nf - i - 1) / rate;
                std::cout << "\r[Partition]   Inserted " << (i + 1) << "/" << nf
                          << " faces (" << (int)bsp.cells.size() << " cells, "
                          << std::fixed << std::setprecision(1)
                          << elapsed << "s elapsed, ~" << eta << "s remaining)"
                          << std::flush;
            }
        }
        std::cout << std::defaultfloat << std::setprecision(6) << std::endl;
    }
    std::cout << "[Partition]   BSP: " << bsp.cells.size() << " cells, "
              << bsp.vertices.size() << " vertices after face insertion" << std::endl;

    // ========================================
    // Insert additional UV boundary split faces (§3.4)
    // For co-planar faces sharing an edge with discontinuous UV,
    // insert an arbitrary face through the shared edge to force the BSP
    // to preserve that edge in the output.
    // ========================================
    if (mesh.FTC.rows() == nf && mesh.TC.rows() > 0)
    {
        // Build edge → adjacent face list
        std::map<std::pair<int, int>, std::vector<int>> edge_faces;
        for (int f = 0; f < nf; ++f)
        {
            for (int e = 0; e < 3; ++e)
            {
                int v0 = mesh.F(f, e), v1 = mesh.F(f, (e + 1) % 3);
                edge_faces[std::make_pair(std::min(v0, v1), std::max(v0, v1))].push_back(f);
            }
        }

        int uv_splits = 0;
        int next_face_id = nf;
        for (auto& [edge, faces] : edge_faces)
        {
            if (faces.size() != 2)
                continue;
            int f0 = faces[0], f1 = faces[1];

            // Check if the two faces are co-planar
            Vec3d n0 = mesh.face_normal(f0);
            Vec3d n1 = mesh.face_normal(f1);
            double a0 = n0.norm(), a1 = n1.norm();
            if (a0 < 1e-15 || a1 < 1e-15)
                continue;
            if (n0.cross(n1).squaredNorm() / (a0 * a0 * a1 * a1) > 1e-10)
                continue; // not co-planar

            // Check if UV coordinates differ at the shared edge
            int v0 = edge.first, v1 = edge.second;
            auto find_tc_index = [&](int f, int v) -> int {
                for (int e = 0; e < 3; ++e)
                    if (mesh.F(f, e) == v)
                        return mesh.FTC(f, e);
                return -1;
            };

            int tc00 = find_tc_index(f0, v0), tc01 = find_tc_index(f0, v1);
            int tc10 = find_tc_index(f1, v0), tc11 = find_tc_index(f1, v1);
            if (tc00 < 0 || tc01 < 0 || tc10 < 0 || tc11 < 0)
                continue;

            bool uv_discontinuous =
                (mesh.TC.row(tc00) - mesh.TC.row(tc10)).squaredNorm() > 1e-12 ||
                (mesh.TC.row(tc01) - mesh.TC.row(tc11)).squaredNorm() > 1e-12;

            if (!uv_discontinuous)
                continue;

            // Create a virtual vertex perpendicular to the co-planar faces,
            // positioned at the edge midpoint offset along the perpendicular.
            Vec3d mid = (mesh.V.row(v0).transpose() + mesh.V.row(v1).transpose()) / 2.0;
            Vec3d edge_dir = (mesh.V.row(v1) - mesh.V.row(v0)).transpose();
            Vec3d perp = edge_dir.cross(n0);
            double perp_len = perp.norm();
            if (perp_len < 1e-15)
                continue;
            perp /= perp_len;

            double offset = edge_dir.norm() * 0.5;
            Vec3d virtual_pt = mid + offset * perp;

            EPoint3 ep(virtual_pt.x(), virtual_pt.y(), virtual_pt.z());
            int new_vi = bsp.add_vertex(ep);
            bsp.insert_face(next_face_id++, v0, v1, new_vi);
            ++uv_splits;
        }
        if (uv_splits > 0)
        {
            std::cout << "[Partition]   Inserted " << uv_splits
                      << " UV boundary split faces" << std::endl;
        }
    }

    // ========================================
    // Step 2: Extract partition from BSP
    // ========================================
    std::cout << "[Partition]   Extracting partition..." << std::endl;
    MatXd V_out;
    MatXi F_out;
    MatXi per_face_cells;
    VecXi face_labels;
    int num_cells;

    bsp.extract_partition(V_out, F_out, per_face_cells, face_labels, num_cells);

    // Remove degenerate faces (where 2+ vertices coincide)
    {
        std::vector<int> good;
        for (int i = 0; i < F_out.rows(); ++i)
        {
            if (F_out(i, 0) != F_out(i, 1) &&
                F_out(i, 1) != F_out(i, 2) &&
                F_out(i, 0) != F_out(i, 2))
                good.push_back(i);
        }
        if ((int)good.size() < F_out.rows())
        {
            int removed = F_out.rows() - (int)good.size();
            MatXi FF(good.size(), 3);
            MatXi PPC(good.size(), 2);
            VecXi FL(good.size());
            for (int i = 0; i < (int)good.size(); ++i)
            {
                FF.row(i) = F_out.row(good[i]);
                PPC.row(i) = per_face_cells.row(good[i]);
                FL(i) = face_labels(good[i]);
            }
            F_out = FF;
            per_face_cells = PPC;
            face_labels = FL;
            std::cout << "[Partition]   Removed " << removed
                      << " degenerate faces" << std::endl;
        }
    }

    int nf_out = F_out.rows();

    // ========================================
    // Step 3: Build face_mapping via 3 passes
    // ========================================

    // --- Pass 1: BSP labels ---
    // The BSP tree directly labels partition faces that lie on an input face's
    // supporting plane and were created during that face's insertion.
    VecXi face_mapping(nf_out);
    int mapped_pass1 = 0;
    for (int i = 0; i < nf_out; ++i)
    {
        int label = face_labels(i);
        if (label >= 0 && label < original_face_count)
        {
            face_mapping(i) = label;
            ++mapped_pass1;
        }
        else
        {
            face_mapping(i) = -1;
        }
    }

    // --- Pass 2: Vertex-triple hash ---
    // If all three vertices of a partition face are original input vertices
    // and they form an input face, map accordingly.
    struct TriHash
    {
        size_t operator()(const std::array<int, 3>& a) const
        {
            size_t h = std::hash<int>()(a[0]);
            h ^= std::hash<int>()(a[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(a[2]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<std::array<int, 3>, int, TriHash> input_tri_map;
    for (int f = 0; f < original_face_count && f < nf; ++f)
    {
        std::array<int, 3> key = {mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)};
        std::sort(key.begin(), key.end());
        input_tri_map[key] = f;
    }

    int mapped_pass2 = 0;
    for (int i = 0; i < nf_out; ++i)
    {
        if (face_mapping(i) >= 0)
            continue;
        // Only check faces where all vertices are original (index < nv)
        if (F_out(i, 0) < nv && F_out(i, 1) < nv && F_out(i, 2) < nv)
        {
            std::array<int, 3> key = {F_out(i, 0), F_out(i, 1), F_out(i, 2)};
            std::sort(key.begin(), key.end());
            auto it = input_tri_map.find(key);
            if (it != input_tri_map.end())
            {
                face_mapping(i) = it->second;
                ++mapped_pass2;
            }
        }
    }

    // --- Pass 3: Barycenter containment ---
    // For remaining unmapped faces, check if the barycenter lies exactly on
    // (within tolerance of) an original input face.
    // Build per-face AABBs for the input mesh to speed up the search.
    struct AABB2
    {
        double lo[3], hi[3];
    };
    std::vector<AABB2> input_aabbs(original_face_count);
    for (int f = 0; f < original_face_count && f < nf; ++f)
    {
        Vec3d a = mesh.V.row(mesh.F(f, 0)).transpose();
        Vec3d b = mesh.V.row(mesh.F(f, 1)).transpose();
        Vec3d c = mesh.V.row(mesh.F(f, 2)).transpose();
        for (int d = 0; d < 3; ++d)
        {
            input_aabbs[f].lo[d] = std::min({a[d], b[d], c[d]});
            input_aabbs[f].hi[d] = std::max({a[d], b[d], c[d]});
        }
    }

    double diag = mesh.bbox_diagonal();
    double plane_tol = diag * 1e-8;
    double bary_tol = 1e-4;

    int mapped_pass3 = 0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:mapped_pass3)
    for (int i = 0; i < nf_out; ++i)
    {
        if (face_mapping(i) >= 0)
            continue;

        Vec3d bc = (V_out.row(F_out(i, 0)) + V_out.row(F_out(i, 1)) +
                    V_out.row(F_out(i, 2)))
                       .transpose() /
                   3.0;

        for (int f = 0; f < original_face_count && f < nf; ++f)
        {
            // AABB check
            bool skip = false;
            for (int d = 0; d < 3; ++d)
            {
                if (bc[d] < input_aabbs[f].lo[d] - plane_tol ||
                    bc[d] > input_aabbs[f].hi[d] + plane_tol)
                {
                    skip = true;
                    break;
                }
            }
            if (skip)
                continue;

            Vec3d a = mesh.V.row(mesh.F(f, 0)).transpose();
            Vec3d b = mesh.V.row(mesh.F(f, 1)).transpose();
            Vec3d c = mesh.V.row(mesh.F(f, 2)).transpose();

            if (!point_on_plane(bc, a, b, c, plane_tol))
                continue;

            if (point_in_triangle_3d(bc, a, b, c, bary_tol))
            {
                face_mapping(i) = f;
                ++mapped_pass3;
                break;
            }
        }
    }

    // ========================================
    // Step 4: Propagate offset_source
    // ========================================
    VecXi offset_source(nf_out);
    for (int i = 0; i < nf_out; ++i)
    {
        int fm = face_mapping(i);
        if (fm >= 0 && fm < mesh.offset_source.size() && mesh.offset_source(fm) >= 0)
            offset_source(i) = mesh.offset_source(fm);
        else
            offset_source(i) = -1;
    }

    // ========================================
    // Step 5: Remove unreferenced vertices
    // ========================================
    {
        MatXd NV;
        MatXi NF;
        VecXi I;
        igl::remove_unreferenced(V_out, F_out, NV, NF, I);
        V_out = NV;
        F_out = NF;
    }

    // ========================================
    // Build PartitionResult
    // ========================================
    PartitionResult result;
    result.mesh.V = V_out;
    result.mesh.F = F_out;
    result.mesh.face_mapping = face_mapping;
    result.mesh.offset_source = offset_source;
    result.per_patch_cells = per_face_cells;
    result.num_cells = num_cells;

    int total_mapped = mapped_pass1 + mapped_pass2 + mapped_pass3;
    int extra_count = nf_out - total_mapped;
    std::cout << "[Partition]   Partition: " << V_out.rows() << " vertices, "
              << nf_out << " faces, " << num_cells << " cells" << std::endl;
    std::cout << "[Partition]   face_mapping: " << total_mapped << " mapped ("
              << "pass1=" << mapped_pass1 << " pass2=" << mapped_pass2
              << " pass3=" << mapped_pass3 << "), " << extra_count << " extra"
              << std::endl;

    return result;
}

} // namespace vpmr
