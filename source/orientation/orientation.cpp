#include "orientation.h"
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace vpmr
{

using Edge = std::pair<int, int>;

static Edge make_edge(int a, int b)
{
    return a < b ? Edge{a, b} : Edge{b, a};
}

static void remove_duplicate_faces(Mesh& mesh, MatXi* per_face_cells)
{
    int nf = mesh.F.rows();
    std::map<std::tuple<int, int, int>, int> face_set;
    std::vector keep(nf, true);

    for (int f = 0; f < nf; ++f)
    {
        int v[3] = {mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)};
        std::sort(v, v + 3);
        auto key = std::make_tuple(v[0], v[1], v[2]);
        auto it = face_set.find(key);
        if (it != face_set.end())
        {
            keep[f] = false;
            keep[it->second] = false;
        }
        else
        {
            face_set[key] = f;
        }
    }

    int count = 0;
    for (int f = 0; f < nf; ++f)
    {
        if (keep[f])
            ++count;
    }

    MatXi newF(count, 3);
    VecXd newVis(count), newOri(count), newOpen(count);
    VecXi newFaceMapping, newOffsetSource;
    MatXi newPerFaceCells;
    if (mesh.face_mapping.size() == nf)
        newFaceMapping.resize(count);
    if (mesh.offset_source.size() == nf)
        newOffsetSource.resize(count);
    if (per_face_cells != nullptr && per_face_cells->rows() == nf)
        newPerFaceCells.resize(count, per_face_cells->cols());
    int idx = 0;
    for (int f = 0; f < nf; ++f)
    {
        if (keep[f])
        {
            newF.row(idx) = mesh.F.row(f);
            if (mesh.visibility.size() > 0)
                newVis(idx) = mesh.visibility(f);
            if (mesh.orientation.size() > 0)
                newOri(idx) = mesh.orientation(f);
            if (mesh.openness.size() > 0)
                newOpen(idx) = mesh.openness(f);
            if (newFaceMapping.size() > 0)
                newFaceMapping(idx) = mesh.face_mapping(f);
            if (newOffsetSource.size() > 0)
                newOffsetSource(idx) = mesh.offset_source(f);
            if (newPerFaceCells.size() > 0)
                newPerFaceCells.row(idx) = per_face_cells->row(f);
            ++idx;
        }
    }
    mesh.F = newF;
    mesh.visibility = newVis;
    mesh.orientation = newOri;
    mesh.openness = newOpen;
    if (newFaceMapping.size() > 0)
        mesh.face_mapping = newFaceMapping;
    if (newOffsetSource.size() > 0)
        mesh.offset_source = newOffsetSource;
    if (newPerFaceCells.size() > 0)
        *per_face_cells = newPerFaceCells;
}

void adjust_orientation(Mesh& mesh, MatXi* per_face_cells)
{
    // Only remove duplicate faces for the initial orientation pass (§3.2).
    // For partition meshes (§3.5.2, per_face_cells != nullptr), duplicate faces
    // have distinct cell adjacency and must not be removed.
    if (per_face_cells == nullptr)
        remove_duplicate_faces(mesh, per_face_cells);

    int nf = mesh.F.rows();
    if (nf == 0)
        return;

    // Build oriented edge → face adjacency
    // For edge (a,b) in face f, orientation is consistent if neighbor has (b,a)
    std::map<Edge, std::vector<std::pair<int, int>>> oriented_edge_faces; // edge → [(face, local_edge_idx)]
    std::map<Edge, std::vector<int>> edge_faces;

    for (int f = 0; f < nf; ++f)
    {
        for (int e = 0; e < 3; ++e)
        {
            int a = mesh.F(f, e);
            int b = mesh.F(f, (e + 1) % 3);
            Edge oe = {a, b};
            oriented_edge_faces[oe].push_back({f, e});
            edge_faces[make_edge(a, b)].push_back(f);
        }
    }

    // Flood-fill patches: connected faces with consistent orientation
    std::vector patch_id(nf, -1);
    int num_patches = 0;

    for (int seed = 0; seed < nf; ++seed)
    {
        if (patch_id[seed] >= 0)
            continue;
        int pid = num_patches++;
        patch_id[seed] = pid;
        std::queue<int> q;
        q.push(seed);

        while (!q.empty())
        {
            int f = q.front();
            q.pop();

            for (int e = 0; e < 3; ++e)
            {
                int a = mesh.F(f, e);
                int b = mesh.F(f, (e + 1) % 3);
                if (edge_faces[make_edge(a, b)].size() != 2)
                    continue;
                // Consistent neighbor shares edge (b,a)
                Edge rev = {b, a};
                auto it = oriented_edge_faces.find(rev);
                if (it == oriented_edge_faces.end())
                    continue;
                for (auto& [nf2, ne] : it->second)
                {
                    if (patch_id[nf2] < 0)
                    {
                        patch_id[nf2] = pid;
                        q.push(nf2);
                    }
                }
            }
        }
    }

    // Per-patch: area-weighted Φ_orientation (Eq. 5)
    std::vector<double> patch_orient_num(num_patches, 0);
    std::vector<double> patch_orient_den(num_patches, 0);

    for (int f = 0; f < nf; ++f)
    {
        // Eq. 5: only visible faces contribute to the weighted average
        bool visible = mesh.visibility.size() > 0 && mesh.visibility(f) > 0.5;
        if (!visible)
            continue;
        double area = mesh.face_area(f);
        double ori = mesh.orientation.size() > 0 ? mesh.orientation(f) : 0;
        patch_orient_num[patch_id[f]] += ori * area;
        patch_orient_den[patch_id[f]] += area;
    }

    // Flip patches with negative weighted orientation
    for (int f = 0; f < nf; ++f)
    {
        int pid = patch_id[f];
        if (patch_orient_den[pid] > 0)
        {
            double weighted_ori = patch_orient_num[pid] / patch_orient_den[pid];
            if (weighted_ori < 0)
            {
                std::swap(mesh.F(f, 1), mesh.F(f, 2));
                if (mesh.orientation.size() > 0)
                    mesh.orientation(f) = -mesh.orientation(f);
                if (per_face_cells != nullptr && per_face_cells->rows() == nf)
                    std::swap((*per_face_cells)(f, 0), (*per_face_cells)(f, 1));
            }
        }
    }
}

} // namespace vpmr
