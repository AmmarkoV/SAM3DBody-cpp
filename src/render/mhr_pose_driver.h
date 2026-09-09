#pragma once
// mhr_pose_driver.h
// Per-frame bridge between MHR inference output and the TRI rendering model.
//
// Two responsibilities:
//   1. Copy pred_vertices into the TRI model's vertex buffer (no skinning needed;
//      the body_model.onnx already outputs the fully-deformed mesh).
//   2. Build OpenGL projection + view matrices from the MHR camera parameters
//      so the rendered mesh aligns with the source image.

#include <string.h>  // memcpy
#include <math.h>    // sqrtf
#include <vector>
#include <algorithm> // std::find
#include "../GraphicsEngine/ModelLoader/model_loader_tri.h"

// Total number of vertices and floats in the MHR body mesh.
#define MHR_VERTEX_COUNT  18439
#define MHR_VERTEX_FLOATS (MHR_VERTEX_COUNT * 3)

// ---------------------------------------------------------------------------
// mhr_update_mesh_vertices
//
// Overwrites the TRI model's vertex buffer with the current frame's
// pred_vertices.  Call this every frame before uploading to the GPU.
//
// pred_vertices : float array of MHR_VERTEX_FLOATS elements, layout [x0,y0,z0, x1,...]
//                 This is MHRResult::pred_vertices.data() from the C++ pipeline.
//
// Note: MHR applies a Y/Z sign flip after skinning (verts[Y,Z] *= -1) to convert
// from the model's internal coordinate system to a camera-facing one.  The C++
// pipeline should replicate that flip.  If the mesh appears mirrored along Y or Z
// during rendering, toggle the flip in the pipeline and retest.
// ---------------------------------------------------------------------------
static inline void mhr_update_mesh_vertices(struct TRI_Model   *model,
        const float        *pred_vertices)
{
    if (!model || !model->vertices || !pred_vertices)
    {
        return;
    }
    memcpy(model->vertices, pred_vertices, MHR_VERTEX_FLOATS * sizeof(float));
}

// ---------------------------------------------------------------------------
// mhr_update_mesh_normals
//
// Recomputes smooth per-vertex normals from the CURRENT (posed) vertex
// positions, replacing the T-pose normals baked into the .tri file at load
// time.  Call this every frame, right after mhr_update_mesh_vertices.
//
// Why this is needed: the mesh is skinned on the CPU (mhr_lbs_compute) and
// only the deformed POSITIONS are re-uploaded per frame — the normal buffer
// was left as GL_STATIC_DRAW T-pose data.  In anything but a near-T-pose
// stance (e.g. a dance clip with raised/twisted limbs), a limb's actual
// orientation has rotated away from its rest-pose orientation but its
// normals haven't, so the lit/shadowed split no longer follows the real
// surface curvature.  It instead falls in whatever place the *stale*
// rest-pose normal happens to cross the N.L=0 threshold, which cuts across
// the limb at an arbitrary angle — seen as sharp, wrong-looking shading
// bands/blocks that don't track the body as it moves.
//
// Cost: one cross product per triangle (~36874 tris) plus a normalize per
// vertex (~18439 verts) — well under a millisecond, negligible next to the
// per-frame network inference cost.
// ---------------------------------------------------------------------------
static inline void mhr_update_mesh_normals(struct TRI_Model *model)
{
    if (!model || !model->vertices || !model->normal || !model->indices)
    {
        return;
    }

    const unsigned int numVerts = model->header.numberOfVertices / 3;
    const unsigned int numTris  = model->header.numberOfIndices / 3;

    memset(model->normal, 0, (size_t)numVerts * 3 * sizeof(float));

    const float       *v = model->vertices;
    float              *n = model->normal;
    const unsigned int *idx = model->indices;

    for (unsigned int t = 0; t < numTris; ++t)
    {
        unsigned int i0 = idx[t*3+0], i1 = idx[t*3+1], i2 = idx[t*3+2];
        const float *p0 = v + i0*3, *p1 = v + i1*3, *p2 = v + i2*3;

        float e1x = p1[0]-p0[0], e1y = p1[1]-p0[1], e1z = p1[2]-p0[2];
        float e2x = p2[0]-p0[0], e2y = p2[1]-p0[1], e2z = p2[2]-p0[2];

        // Unnormalized cross product: magnitude ~ 2x triangle area, so
        // bigger triangles naturally contribute more to their vertices'
        // averaged normal (standard area-weighted vertex normal).
        float fx = e1y*e2z - e1z*e2y;
        float fy = e1z*e2x - e1x*e2z;
        float fz = e1x*e2y - e1y*e2x;

        n[i0*3+0] += fx; n[i0*3+1] += fy; n[i0*3+2] += fz;
        n[i1*3+0] += fx; n[i1*3+1] += fy; n[i1*3+2] += fz;
        n[i2*3+0] += fx; n[i2*3+1] += fy; n[i2*3+2] += fz;
    }

    for (unsigned int i = 0; i < numVerts; ++i)
    {
        float x = n[i*3+0], y = n[i*3+1], z = n[i*3+2];
        float len = sqrtf(x*x + y*y + z*z);
        if (len > 1e-12f)
        {
            n[i*3+0] = x / len;
            n[i*3+1] = y / len;
            n[i*3+2] = z / len;
        }
    }
}

// ---------------------------------------------------------------------------
// MHR_VertexAdjacency / mhr_build_vertex_adjacency / mhr_smooth_mesh_normals
//
// The per-vertex normals from mhr_update_mesh_normals() are already a
// correct area-weighted average of each vertex's OWN adjacent faces — but on
// a body-sized mesh a cylindrical part (forearm, calf, thigh) only has ~12-18
// sides around its circumference, so each ring of triangles still turns the
// normal by a visible number of degrees from one triangle to the next.  The
// eye reads that as "fluting": a series of parallel light/dark stripes
// running down the limb, distinct from the earlier stale-pose banding bug.
//
// The standard fix (as used by e.g. MeshLab's "smooth vertex normals") is to
// widen the averaging beyond one vertex's immediate faces to its 1-ring
// topological neighbours, blurring the normal field just enough to hide the
// facets while a genuinely low-poly silhouette/outline is untouched (only
// normals are smoothed, not positions).  The mesh topology (indices) never
// changes frame to frame, so the adjacency list is built ONCE at load time;
// only the O(edges) averaging pass repeats per frame (~110K int/float ops
// for 2 iterations over this mesh — negligible next to inference cost).
// ---------------------------------------------------------------------------
struct MHR_VertexAdjacency
{
    std::vector<unsigned int> offsets;    // size numVerts+1
    std::vector<unsigned int> neighbors;  // size offsets[numVerts]
};

static inline void mhr_build_vertex_adjacency(const struct TRI_Model *model,
                                               MHR_VertexAdjacency     &adj)
{
    const unsigned int numVerts = model->header.numberOfVertices / 3;
    const unsigned int numTris  = model->header.numberOfIndices / 3;

    std::vector<std::vector<unsigned int>> tmp(numVerts);
    auto addEdge = [&](unsigned int a, unsigned int b) {
        std::vector<unsigned int> &v = tmp[a];
        if (std::find(v.begin(), v.end(), b) == v.end()) v.push_back(b);
    };
    for (unsigned int t = 0; t < numTris; ++t)
    {
        unsigned int i0 = model->indices[t*3+0];
        unsigned int i1 = model->indices[t*3+1];
        unsigned int i2 = model->indices[t*3+2];
        addEdge(i0, i1); addEdge(i1, i0);
        addEdge(i1, i2); addEdge(i2, i1);
        addEdge(i2, i0); addEdge(i0, i2);
    }

    adj.offsets.assign(numVerts + 1, 0);
    unsigned int total = 0;
    for (unsigned int v = 0; v < numVerts; ++v) { adj.offsets[v] = total; total += (unsigned int)tmp[v].size(); }
    adj.offsets[numVerts] = total;

    adj.neighbors.resize(total);
    for (unsigned int v = 0; v < numVerts; ++v)
        std::copy(tmp[v].begin(), tmp[v].end(), adj.neighbors.begin() + adj.offsets[v]);
}

static inline void mhr_smooth_mesh_normals(struct TRI_Model           *model,
                                            const MHR_VertexAdjacency  &adj,
                                            std::vector<float>         &scratch,
                                            int                         iterations)
{
    const unsigned int numVerts = model->header.numberOfVertices / 3;
    scratch.resize((size_t)numVerts * 3);
    float *n = model->normal;

    for (int it = 0; it < iterations; ++it)
    {
        memcpy(scratch.data(), n, (size_t)numVerts * 3 * sizeof(float));
        for (unsigned int v = 0; v < numVerts; ++v)
        {
            unsigned int b = adj.offsets[v], e = adj.offsets[v+1];
            float sx = scratch[v*3+0], sy = scratch[v*3+1], sz = scratch[v*3+2];
            for (unsigned int k = b; k < e; ++k)
            {
                unsigned int nb = adj.neighbors[k];
                sx += scratch[nb*3+0]; sy += scratch[nb*3+1]; sz += scratch[nb*3+2];
            }
            float len = sqrtf(sx*sx + sy*sy + sz*sz);
            if (len > 1e-12f)
            {
                n[v*3+0] = sx / len;
                n[v*3+1] = sy / len;
                n[v*3+2] = sz / len;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mhr_camera_matrices
//
// Builds column-major OpenGL matrices from the MHR pinhole camera parameters.
//
// out_proj  [16] : GL projection matrix (column-major, right-handed NDC)
// out_view  [16] : GL model-view matrix (pure translation by pred_cam_t)
// focal_length   : MHRResult::focal_length (pixels)
// pred_cam_t [3] : MHRResult::pred_cam_t  (tx, ty, tz in metres)
// img_w, img_h   : source image dimensions (pixels)
// ---------------------------------------------------------------------------
static inline void mhr_camera_matrices(float       out_proj[16],
                                       float       out_view[16],
                                       float       focal_length,
                                       const float pred_cam_t[3],
                                       int         img_w,
                                       int         img_h)
{
    const float near_plane = 0.01f;
    const float far_plane  = 100.0f;

    // Standard OpenGL perspective from pinhole focal length (pixels).
    // Principal point assumed at image centre (MHR default).
    float p00 = 2.0f * focal_length / (float)img_w;
    float p11 = 2.0f * focal_length / (float)img_h;
    float p22 = -(far_plane + near_plane) / (far_plane - near_plane);
    float p32 = -2.0f * far_plane * near_plane / (far_plane - near_plane);

    // Column-major layout: out[col*4 + row]
    out_proj[ 0] = p00;
    out_proj[ 4] = 0.0f;
    out_proj[ 8] = 0.0f;
    out_proj[12] = 0.0f;
    out_proj[ 1] = 0.0f;
    out_proj[ 5] = p11;
    out_proj[ 9] = 0.0f;
    out_proj[13] = 0.0f;
    out_proj[ 2] = 0.0f;
    out_proj[ 6] = 0.0f;
    out_proj[10] = p22;
    out_proj[14] = p32;
    out_proj[ 3] = 0.0f;
    out_proj[ 7] = 0.0f;
    out_proj[11] = -1.0f;
    out_proj[15] = 0.0f;

    // Model-view matrix that replicates the Python pyrender pipeline:
    //   Python:   verts[Y,Z] *= -1  (LBS flip)  then
    //             renderer applies 180° rotation around X  (flips Y,Z again → net identity on verts)
    //             camera placed at (-tx, ty, tz) looking in -Z
    //   OpenGL:   C++ LBS already has the [Y,Z]*=-1 flip applied to verts, so we
    //             need to undo it with another Y,Z sign flip on the diagonal (-1,-1),
    //             then translate by pred_cam_t matching "verts + pred_cam_t" in Python.
    //   Result:   v_view = (x+tx, -(−y)−ty, -(−z)−tz) = (x+tx, y−ty, z−tz)
    //             which matches pyrender camera space exactly.
    out_view[ 0] = 1.0f;
    out_view[ 4] = 0.0f;
    out_view[ 8] = 0.0f;
    out_view[12] =  pred_cam_t[0];
    out_view[ 1] = 0.0f;
    out_view[ 5] = -1.0f;
    out_view[ 9] = 0.0f;
    out_view[13] = -pred_cam_t[1];
    out_view[ 2] = 0.0f;
    out_view[ 6] = 0.0f;
    out_view[10] = -1.0f;
    out_view[14] = -pred_cam_t[2];
    out_view[ 3] = 0.0f;
    out_view[ 7] = 0.0f;
    out_view[11] = 0.0f;
    out_view[15] = 1.0f;
}
