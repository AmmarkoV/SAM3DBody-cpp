// arf_writer.cpp — see arf_writer.h and ARF.md for the container/JSON subset
// implemented and its deviations from the spec's own examples.

#include "arf_writer.h"
#include "arf_json.h"
#include "bbox_iou.h"
#include "fast_sam_3dbody.h"
#include "mhr_joint_table.h"

extern "C" {
#include "ModelLoader/model_loader_transform_joints.h"
#include "ModelLoader/model_loader_tri.h"
}

#include "miniz.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace
{

// ─── small math helpers (self-contained duplicates of bvh_writer.cpp's —
// pure primitives, not FK; see mhr_fk.h for the shared kinematics core) ────

constexpr float TRACK_IOU_THRESH  = 0.10f;
constexpr int   TRACK_MAX_MISSING = 90;   // 3 s at 30 fps, matches BVHWriter

inline bool bbox_looks_valid(const std::array<float,4>& b)
{
    const float w = b[2] - b[0];
    const float h = b[3] - b[1];
    if (w < 8.f || h < 8.f) return false;
    if (b[0] == 0.f && b[1] == 0.f) return false;
    return true;
}

// XYZW quaternion -> row-major 3x3.
inline void quat_to_mat3(const float* q, float m[9])
{
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    m[0]=1-2*(yy+zz); m[1]=2*(xy-wz);   m[2]=2*(xz+wy);
    m[3]=2*(xy+wz);   m[4]=1-2*(xx+zz); m[5]=2*(yz-wx);
    m[6]=2*(xz-wy);   m[7]=2*(yz+wx);   m[8]=1-2*(xx+yy);
}

// Row-major 4x4 TRS composition: M = T(t) * R(q) * S(s).
inline void compose_trs_mat4(const float t[3], const float q[4], float s, float m[16])
{
    float r[9]; quat_to_mat3(q, r);
    m[0]=r[0]*s; m[1]=r[1]*s; m[2]=r[2]*s;  m[3]=t[0];
    m[4]=r[3]*s; m[5]=r[4]*s; m[6]=r[5]*s;  m[7]=t[1];
    m[8]=r[6]*s; m[9]=r[7]*s; m[10]=r[8]*s; m[11]=t[2];
    m[12]=0.f;   m[13]=0.f;   m[14]=0.f;    m[15]=1.f;
}

inline float median_of(std::vector<float> v)
{
    if (v.empty()) return 0.f;
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

// Median of a [x0,y0,z0, x1,y1,z1, ...] sample list, per axis.
void median_vec3(const std::vector<float>& samples, float out[3])
{
    if (samples.size() < 3) { out[0]=out[1]=out[2]=0.f; return; }
    const size_t n = samples.size() / 3;
    std::vector<float> axis(n);
    for (int a = 0; a < 3; ++a)
    {
        for (size_t i = 0; i < n; ++i) axis[i] = samples[i*3 + a];
        out[a] = median_of(axis);
    }
}

// ─── byte-buffer helpers for our binary data items / AAU stream ───────────
using Bytes = std::vector<uint8_t>;

template <typename T>
inline void put(Bytes& b, T v)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
inline void put_bytes(Bytes& b, const void* data, size_t n)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    b.insert(b.end(), p, p + n);
}
// ─── big-endian primitives, for the AAU animation stream only ──────────────
// Every other binary payload here (dense/sparse tensors, GLB) stays little-
// endian, this project's own convention — but the spec's AAU bitstream tables
// use "uimsbf" (unsigned integer, most significant bit first), the same
// MPEG-systems convention ISOBMFF/MPEG-2 Systems use, where it means
// big-endian. See ARF.md "Avatar Animation Units".
inline void put_u16be(Bytes& b, uint16_t v)
{
    b.push_back((uint8_t)(v >> 8));
    b.push_back((uint8_t)(v & 0xFF));
}
inline void put_u32be(Bytes& b, uint32_t v)
{
    b.push_back((uint8_t)(v >> 24));
    b.push_back((uint8_t)(v >> 16));
    b.push_back((uint8_t)(v >> 8));
    b.push_back((uint8_t)(v & 0xFF));
}
inline void put_f32be(Bytes& b, float v)
{
    uint32_t bits; std::memcpy(&bits, &v, 4);
    put_u32be(b, bits);
}
inline void put_floats_be(Bytes& b, const float* v, size_t n)
{
    for (size_t i = 0; i < n; ++i) put_f32be(b, v[i]);
}
// 8-bit length prefix — the AAU_CONFIG profile string's own encoding, distinct
// from this format's other (32-bit-length-prefixed) strings; no NUL.
inline void put_str8(Bytes& b, const std::string& s)
{
    put<uint8_t>(b, (uint8_t)s.size());
    put_bytes(b, s.data(), s.size());
}

// glTF 2.0 accessor component type codes (application/mpeg.arf.{dense,sparse}
// tensors declare their element type this way, per ARF.md).
constexpr int32_t GLTF_FLOAT        = 5126;
constexpr int32_t GLTF_UNSIGNED_INT = 5125;

// Dense tensor payload: num_of_dims, dims[], dtype, then the raw row-major
// data block (see ARF.md "Dense tensors").
Bytes build_dense_tensor(const std::vector<int32_t>& dims, int32_t dtype,
                         const void* data, size_t byte_size)
{
    Bytes b;
    put<int32_t>(b, (int32_t)dims.size());
    for (int32_t d : dims) put<int32_t>(b, d);
    put<int32_t>(b, dtype);
    put_bytes(b, data, byte_size);
    return b;
}

// Sparse tensor payload: num_of_dims, dims[], valueCount, itype, dtype, then
// a flat row-major index per nonzero (itype) followed by its value (dtype).
// See ARF.md "Sparse tensors" — used for the skin weights.
Bytes build_sparse_tensor(const std::vector<int32_t>& dims,
                          const std::vector<uint32_t>& flat_indices,
                          const std::vector<float>& values)
{
    Bytes b;
    put<int32_t>(b, (int32_t)dims.size());
    for (int32_t d : dims) put<int32_t>(b, d);
    put<int32_t>(b, (int32_t)flat_indices.size());
    put<int32_t>(b, GLTF_UNSIGNED_INT);
    put<int32_t>(b, GLTF_FLOAT);
    put_bytes(b, flat_indices.data(), flat_indices.size() * sizeof(uint32_t));
    put_bytes(b, values.data(), values.size() * sizeof(float));
    return b;
}

// ─── minimal GLB (binary glTF) encoder — for BlendshapeSet.shapes[i] ───────
// Per spec each blendshape target is its own GLB: one mesh, one primitive, a
// POSITION accessor (float32 VEC3) and an indices accessor (uint32 SCALAR),
// no materials/textures. GLB is little-endian throughout (unlike the AAU
// stream above) — see ARF.md "Container" and the BlendshapeSet.shapes note.
namespace glb
{
constexpr uint32_t MAGIC      = 0x46546C67u;  // "glTF"
constexpr uint32_t VERSION    = 2u;
constexpr uint32_t CHUNK_JSON = 0x4E4F534Au;  // "JSON"
constexpr uint32_t CHUNK_BIN  = 0x004E4942u;  // "BIN\0"
constexpr int32_t  TARGET_ARRAY = 34962;      // ARRAY_BUFFER
constexpr int32_t  TARGET_INDEX = 34963;      // ELEMENT_ARRAY_BUFFER

inline void text(std::string& s, const char* t) { s += t; }
inline void num(std::string& s, unsigned long long v) { s += std::to_string(v); }
inline void flt(std::string& s, float v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", (double)v);
    s += buf;
}
} // namespace glb

// One GLB per blendshape target: `positions` holds that shape's *absolute*
// deformed vertex positions (base mesh + delta — see the BlendshapeSet.shapes
// note in ARF.md), `indices` is the same topology as the base mesh.
// `colors` (RGB 0-1 per vertex) adds a glTF COLOR_0 attribute — used for the
// --skin-color TextureSet material, never for blendshapes.
Bytes build_glb_mesh(const std::vector<float>& positions, uint32_t n_verts,
                     const std::vector<uint32_t>& indices, uint32_t n_tris,
                     const std::vector<float>* colors = nullptr)
{
    using namespace glb;
    const size_t positions_bytes = (size_t)n_verts * 3 * sizeof(float);
    const size_t indices_bytes   = (size_t)n_tris * 3 * sizeof(uint32_t);
    const size_t colors_bytes    = colors ? (size_t)n_verts * 3 * sizeof(float) : 0;

    float mn[3] = {  1e30f,  1e30f,  1e30f };
    float mx[3] = { -1e30f, -1e30f, -1e30f };
    for (uint32_t v = 0; v < n_verts; ++v)
        for (int c = 0; c < 3; ++c)
        {
            float val = positions[(size_t)v*3 + c];
            mn[c] = std::min(mn[c], val);
            mx[c] = std::max(mx[c], val);
        }

    std::string json;
    text(json, "{\"asset\":{\"version\":\"2.0\"},");
    text(json, "\"buffers\":[{\"byteLength\":");
    num(json, (unsigned long long)(positions_bytes + indices_bytes + colors_bytes));
    text(json, "}],");
    text(json, "\"bufferViews\":[");
    text(json, "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":");
    num(json, (unsigned long long)positions_bytes);
    text(json, ",\"target\":"); num(json, TARGET_ARRAY); text(json, "},");
    text(json, "{\"buffer\":0,\"byteOffset\":");
    num(json, (unsigned long long)positions_bytes);
    text(json, ",\"byteLength\":");
    num(json, (unsigned long long)indices_bytes);
    text(json, ",\"target\":"); num(json, TARGET_INDEX); text(json, "}");
    if (colors)
    {
        text(json, ",{\"buffer\":0,\"byteOffset\":");
        num(json, (unsigned long long)(positions_bytes + indices_bytes));
        text(json, ",\"byteLength\":");
        num(json, (unsigned long long)colors_bytes);
        text(json, ",\"target\":"); num(json, TARGET_ARRAY); text(json, "}");
    }
    text(json, "],");
    text(json, "\"accessors\":[");
    text(json, "{\"bufferView\":0,\"componentType\":"); num(json, GLTF_FLOAT);
    text(json, ",\"count\":"); num(json, n_verts);
    text(json, ",\"type\":\"VEC3\",\"min\":[");
    for (int c = 0; c < 3; ++c) { if (c) text(json, ","); flt(json, mn[c]); }
    text(json, "],\"max\":[");
    for (int c = 0; c < 3; ++c) { if (c) text(json, ","); flt(json, mx[c]); }
    text(json, "]},");
    text(json, "{\"bufferView\":1,\"componentType\":"); num(json, GLTF_UNSIGNED_INT);
    text(json, ",\"count\":"); num(json, (unsigned long long)n_tris * 3);
    text(json, ",\"type\":\"SCALAR\"}");
    if (colors)
    {
        text(json, ",{\"bufferView\":2,\"componentType\":"); num(json, GLTF_FLOAT);
        text(json, ",\"count\":"); num(json, n_verts);
        text(json, ",\"type\":\"VEC3\"}");
    }
    text(json, "],");
    text(json, colors ? "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"COLOR_0\":2},\"indices\":1,\"mode\":4}]}]}"
                      : "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1,\"mode\":4}]}]}");

    while (json.size() % 4 != 0) json.push_back(' ');   // pad JSON chunk to 4 bytes

    Bytes bin;
    bin.reserve(positions_bytes + indices_bytes + colors_bytes);
    put_bytes(bin, positions.data(), positions_bytes);
    put_bytes(bin, indices.data(), indices_bytes);
    if (colors) put_bytes(bin, colors->data(), colors_bytes);
    // No padding needed: every source array is already a multiple of 4 bytes.

    const uint32_t total_length = (uint32_t)(12 + 8 + json.size() + 8 + bin.size());

    Bytes out;
    put<uint32_t>(out, glb::MAGIC);
    put<uint32_t>(out, glb::VERSION);
    put<uint32_t>(out, total_length);
    put<uint32_t>(out, (uint32_t)json.size());
    put<uint32_t>(out, glb::CHUNK_JSON);
    put_bytes(out, json.data(), json.size());
    put<uint32_t>(out, (uint32_t)bin.size());
    put<uint32_t>(out, glb::CHUNK_BIN);
    put_bytes(out, bin.data(), bin.size());
    return out;
}

// ─── Avatar Animation Unit (AAU) framing ───────────────────────────────────
// header{(unit_type<<1)|reserved, unit_length BE} + payload{timestamp BE,
// type-specific fields, all big-endian} — see ARF.md "Avatar Animation Units".
constexpr uint8_t AAU_CONFIG     = 0;
constexpr uint8_t AAU_BLENDSHAPE = 1;
constexpr uint8_t AAU_JOINT      = 2;

void append_aau(Bytes& stream, uint8_t type, const Bytes& payload)
{
    put<uint8_t>(stream, (uint8_t)(type << 1));   // reserved bit = 0
    put_u32be(stream, (uint32_t)payload.size());
    stream.insert(stream.end(), payload.begin(), payload.end());
}

Bytes build_config_payload(const std::string& profile, float timescale)
{
    Bytes p;
    put_u32be(p, 0u);            // timestamp: always 0 for the config unit
    put_str8(p, profile);
    put_f32be(p, timescale);
    return p;
}

// joint_set_id is the declared id of components.skeletons[0] (always 0, this
// writer's own convention — see the id-assignment note in dump_one_person).
Bytes build_joint_payload(uint32_t timestamp_ticks, uint16_t joint_set_id,
                          const float* mats, int n_joints)
{
    Bytes p;
    put_u32be(p, timestamp_ticks);
    put_u16be(p, joint_set_id);
    put<uint8_t>(p, 0);                       // velocity_present=0, reserved=0
    put_u16be(p, (uint16_t)(n_joints - 1));   // joint_count_minus1
    for (int j = 0; j < n_joints; ++j)
    {
        put_u16be(p, (uint16_t)j);
        put_floats_be(p, mats + (size_t)j * 16, 16);
    }
    return p;
}

// blendshape_set_id is the declared id of components.blendshapeSets[0]
// (always 0, this writer's own convention).
Bytes build_blendshape_payload(uint32_t timestamp_ticks, uint16_t blendshape_set_id,
                               const float* weights, int n)
{
    Bytes p;
    put_u32be(p, timestamp_ticks);
    put_u16be(p, blendshape_set_id);
    put<uint8_t>(p, 0);                       // confidence_present=0, reserved=0
    put_u16be(p, (uint16_t)(n - 1));          // blendshape_count_minus1
    for (int k = 0; k < n; ++k)
    {
        put_u16be(p, (uint16_t)k);
        put_f32be(p, weights[k]);
    }
    return p;
}

// "<stem>_<prefix><id>.arfz" — same convention as bvh_writer.cpp's
// per_person_path(), different default extension.
std::string per_person_path(const std::string& base, const std::string& prefix, int id)
{
    auto dot   = base.find_last_of('.');
    auto slash = base.find_last_of("/\\");
    bool has_ext = (dot != std::string::npos) && (slash == std::string::npos || dot > slash);
    if (has_ext) return base.substr(0, dot) + "_" + prefix + std::to_string(id) + base.substr(dot);
    return base + "_" + prefix + std::to_string(id) + ".arfz";
}

} // namespace

// ─── tracker (greedy IoU — identical algorithm to BVHWriter's; unrelated to
// BVH, so this is a pure bbox-tracking duplicate, not a second FK) ─────────

std::vector<int> ARFWriter::assign_tracks(const std::vector<fsb::MHRResult>& results)
{
    const int F = session_frames_;
    std::vector<int> result_ids(results.size(), -1);

    struct Pair { int det; int track; float iou; };
    std::vector<Pair> pairs;
    pairs.reserve(results.size() * std::max((size_t)1, tracks_.size()));
    for (size_t d = 0; d < results.size(); ++d)
    {
        const float* db = results[d].bbox.data();
        for (size_t t = 0; t < tracks_.size(); ++t)
        {
            float v = fsb::bbox_iou(db, tracks_[t].bbox);
            if (v >= TRACK_IOU_THRESH) pairs.push_back({(int)d, (int)t, v});
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.iou > b.iou; });

    std::vector<char> det_taken(results.size(), 0);
    std::vector<char> track_taken(tracks_.size(), 0);
    for (const auto& p : pairs)
    {
        if (det_taken[p.det] || track_taken[p.track]) continue;
        det_taken[p.det]     = 1;
        track_taken[p.track] = 1;
        result_ids[p.det]    = tracks_[p.track].id;
        const auto& bb = results[p.det].bbox;
        tracks_[p.track].bbox[0]=bb[0]; tracks_[p.track].bbox[1]=bb[1];
        tracks_[p.track].bbox[2]=bb[2]; tracks_[p.track].bbox[3]=bb[3];
        tracks_[p.track].last_seen_frame = F;
    }
    for (size_t d = 0; d < results.size(); ++d)
    {
        if (det_taken[d]) continue;
        Track t;
        t.id = next_track_id_++;
        t.bbox[0]=results[d].bbox[0]; t.bbox[1]=results[d].bbox[1];
        t.bbox[2]=results[d].bbox[2]; t.bbox[3]=results[d].bbox[3];
        t.last_seen_frame = F;
        tracks_.push_back(t);
        result_ids[d] = t.id;
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [F](const Track& t) { return (F - t.last_seen_frame) > TRACK_MAX_MISSING; }),
                  tracks_.end());
    return result_ids;
}

// ─── open / close / frame accumulation ─────────────────────────────────────

bool ARFWriter::open(const std::string& out_path,
                     const std::string& lbs_path,
                     const std::string& mesh_path,
                     float               frame_time,
                     bool                export_face)
{
    out_path_   = out_path;
    frame_time_ = frame_time;

    if (lbs_path.empty() || !std::ifstream(lbs_path).good())
    {
        fprintf(stderr, "[ARFWriter] cannot read lbs_path '%s'\n", lbs_path.c_str());
        return false;
    }
    lbs_ = mhr_lbs_load(lbs_path.c_str());
    if (!lbs_)
    {
        fprintf(stderr, "[ARFWriter] mhr_lbs_load('%s') failed\n", lbs_path.c_str());
        return false;
    }

    mesh_ = tri_allocateModel();
    if (!mesh_ || !tri_loadModel(mesh_path.c_str(), mesh_))
    {
        fprintf(stderr, "[ARFWriter] tri_loadModel('%s') failed — a base avatar\n"
                        "            needs mesh topology; pass --onnx-dir with\n"
                        "            body_mesh.tri present, or check the path.\n",
                mesh_path.c_str());
        if (mesh_) { tri_freeModel(mesh_); mesh_ = nullptr; }
        mhr_lbs_free(lbs_);
        lbs_ = nullptr;
        return false;
    }
    // NOTE: TRI_Header::numberOfVertices is a flat FLOAT count (3 per vertex —
    // see model_loader_tri.c's malloc/fread sizing), not a vertex count; divide
    // by 3 before comparing against MHR_LBS_Data::n_verts.
    if ((int)(mesh_->header.numberOfVertices / 3) != lbs_->n_verts)
    {
        fprintf(stderr, "[ARFWriter] warning: body_mesh.tri vertex count (%u) != "
                        "body_model.lbs n_verts (%d) — mesh/skin may misalign\n",
                mesh_->header.numberOfVertices / 3, lbs_->n_verts);
    }

    fk_.init(lbs_);
    export_face_ = export_face && lbs_->n_face_pc > 0 && lbs_->face_vectors != nullptr;

    tracks_.clear();
    people_.clear();
    next_track_id_  = 0;
    session_frames_ = 0;

    fprintf(stderr, "[ARFWriter] base avatar ready: %d joints, %d verts, %d shape PCs%s\n",
            lbs_->n_joints, lbs_->n_verts, lbs_->n_shape_pc,
            export_face_ ? ", face blendshapes ON" : "");
    return true;
}

void ARFWriter::append_frame_for(PerPerson& p, const fsb::MHRResult& r)
{
    fk_.compute(r);
    const int nj = fk_.n_joints();
    const auto& q_local = fk_.q_local();
    const auto& jp_all  = fk_.joint_params();

    const size_t base = (size_t)p.frame_count * nj * 16;
    p.joint_mats.resize(base + (size_t)nj * 16);
    for (int j = 0; j < nj; ++j)
    {
        const float* jp  = &jp_all[j * 7];
        const float* off = lbs_->joint_offsets + j * 3;
        float t[3];
        if (lbs_->joint_parents[j] < 0)
        {
            // Root: the PT-decoded translation delta only ever captures a tiny
            // internal wobble — actual world position comes from the camera
            // translation head (pred_cam_t, metres).
            //
            // Y and Z are negated.  The MHR pipeline renders with
            //     display = F · (F · v_model) + F · t,   F = diag(1,-1,-1)
            // (mhr_pose_driver.h: the LBS buffer carries verts[Y,Z] *= -1 and
            // the GL view matrix flips them back while translating by
            // (+tx,-ty,-tz)).  Since F² = I the two vertex flips cancel, leaving
            //     display = v_model + F · t.
            // We export v_model — the unflipped Y-up rest mesh and pose — so the
            // translation that belongs beside it is F·t, not t.  Exporting t
            // unflipped put the body above the camera and behind it in Z.
            // The rotations need no change: F cancels itself on the pose.
            t[0] =  r.pred_cam_t[0] * 100.0f;
            t[1] = -r.pred_cam_t[1] * 100.0f;
            t[2] = -r.pred_cam_t[2] * 100.0f;
        }
        else
        {
            t[0] = off[0]+jp[0]; t[1] = off[1]+jp[1]; t[2] = off[2]+jp[2];
        }
        float s = exp2f(jp[6]);
        compose_trs_mat4(t, &q_local[j*4], s, &p.joint_mats[base + (size_t)j*16]);
    }

    // Rest-local bone-vector samples (direct MHR parent; skips the root).
    for (int j = 0; j < nj; ++j)
    {
        int par = lbs_->joint_parents[j];
        if (par < 0) continue;
        float dv[3];
        fk_.rest_local_bone_vector(j, par, dv);
        auto& vec = p.bone_samples[j];
        vec.push_back(dv[0]); vec.push_back(dv[1]); vec.push_back(dv[2]);
    }

    // Identity-shape accumulation (baked at close(), see ARF.md).
    if (!r.shape.empty())
    {
        if (p.shape_sum.empty()) p.shape_sum.assign(r.shape.size(), 0.0);
        for (size_t k = 0; k < r.shape.size() && k < p.shape_sum.size(); ++k)
            p.shape_sum[k] += r.shape[k];
        ++p.shape_samples;
    }

    if (export_face_)
    {
        const int nfp = lbs_->n_face_pc;
        const size_t fbase = (size_t)p.frame_count * nfp;
        p.face_weights.resize(fbase + nfp);
        if (!r.face_params.empty())
            for (int k = 0; k < nfp && k < (int)r.face_params.size(); ++k)
                p.face_weights[fbase + k] = r.face_params[k];
    }

    ++p.frame_count;
}

void ARFWriter::pad_continuation_frame(PerPerson& p)
{
    if (p.frame_count == 0) return;
    const int nj = fk_.n_joints();
    const float* prev = p.joint_mats.data() + (size_t)(p.frame_count - 1) * nj * 16;
    p.joint_mats.insert(p.joint_mats.end(), prev, prev + (size_t)nj * 16);
    if (export_face_)
    {
        const int nfp = lbs_->n_face_pc;
        const float* fprev = p.face_weights.data() + (size_t)(p.frame_count - 1) * nfp;
        p.face_weights.insert(p.face_weights.end(), fprev, fprev + nfp);
    }
    ++p.frame_count;
}

void ARFWriter::write_frame_external(const std::vector<fsb::MHRResult>& results,
                                     const std::vector<int>& track_ids,
                                     const std::vector<int>& pad_ids)
{
    if (!lbs_) return;
    if (results.size() != track_ids.size())
    {
        fprintf(stderr, "[ARFWriter] write_frame_external: results/track_ids "
                        "size mismatch (%zu vs %zu)\n", results.size(), track_ids.size());
        return;
    }
    const int nj = fk_.n_joints();
    for (size_t d = 0; d < results.size(); ++d)
    {
        int id = track_ids[d];
        if (id < 0) continue;
        PerPerson& p = people_[id];
        if (p.id < 0)
        {
            p.id = id;
            p.frame_count = 0;
            p.bone_samples.assign(nj, std::vector<float>{});
        }
        append_frame_for(p, results[d]);
        if (id >= next_track_id_) next_track_id_ = id + 1;
    }
    for (int id : pad_ids)
    {
        auto it = people_.find(id);
        if (it == people_.end()) continue;
        pad_continuation_frame(it->second);
    }
    ++session_frames_;
}

void ARFWriter::write_frame(const std::vector<fsb::MHRResult>& results)
{
    if (!lbs_) return;
    std::vector<fsb::MHRResult> filtered;
    filtered.reserve(results.size());
    for (const auto& r : results) if (bbox_looks_valid(r.bbox)) filtered.push_back(r);

    std::vector<int> ids = assign_tracks(filtered);
    const int nj = fk_.n_joints();
    for (size_t d = 0; d < filtered.size(); ++d)
    {
        int id = ids[d];
        PerPerson& p = people_[id];
        if (p.id < 0)
        {
            p.id = id;
            p.frame_count = 0;
            p.bone_samples.assign(nj, std::vector<float>{});
        }
        append_frame_for(p, filtered[d]);
    }
    for (const auto& t : tracks_)
    {
        if (t.last_seen_frame == session_frames_) continue;
        auto it = people_.find(t.id);
        if (it == people_.end()) continue;
        pad_continuation_frame(it->second);
    }
    ++session_frames_;
}

// ─── close-time: bake mesh, build arf.json, assemble the .arfz ────────────

// Row-major 4x4: out = a * b.
inline void mul4(const float* a, const float* b, float* out)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r*4+c] = a[r*4]*b[c] + a[r*4+1]*b[4+c] + a[r*4+2]*b[8+c] + a[r*4+3]*b[12+c];
}

// --ground.  The pose is estimated in the capture camera's frame, so with a
// pitched camera the real floor is a sloped plane in the data and the feet
// sink or float as the person moves in depth.  Per frame, skin just the foot
// vertices exactly as a player would (globals * inverse bind * rest mesh) and
// take the lowest one; fit a plane through those points; then rotate/shift
// the root so that plane is Y = 0.  That rigid transform removes the camera
// pitch; per-frame depth noise still bobs the body up and down (a 20 cm depth
// error is ~7 cm of height for a camera looking down), so a second pass then
// shifts the root vertically per frame to put the planted foot on the floor.
// Only the root's matrices change.
//   * Tilt is fitted only along horizontal directions the feet travel at
//     least GROUND_MIN_TRAVEL_CM (1 std); elsewhere it is unmeasurable, and
//     a person standing in place is only dropped onto the floor.
//   * Frames with a foot in the air are trimmed: GROUND_ITERS rounds, each
//     keeping the GROUND_KEEP lowest residuals.
//   * A fitted tilt above GROUND_MAX_TILT_DEG is taken as noise and ignored.
//   * A foot is planted while its lowest point moves slower than
//     GROUND_PLANT_CMS horizontally (centred over +-GROUND_SPEED_HALF
//     frames); in a frame with a planted foot the lower foot (a sliding
//     step can be lower than the planted one) is put at Y = 0; frames with no
//     planted foot interpolate the shift between their neighbours, but
//     never so far that a foot ends up below the floor.
void ARFWriter::ground_person(PerPerson& p, const std::vector<float>& rest_verts)
{
    static constexpr float GROUND_MIN_TRAVEL_CM = 10.f;
    static constexpr float GROUND_KEEP          = 0.7f;
    static constexpr int   GROUND_ITERS         = 5;
    static constexpr float GROUND_MAX_TILT_DEG  = 30.f;
    static constexpr float GROUND_PLANT_CMS     = 50.f;
    static constexpr int   GROUND_SPEED_HALF    = 2;
    static constexpr int   GROUND_SMOOTH_HALF   = 2;

    const int nj = lbs_->n_joints, nv = lbs_->n_verts, nf = p.frame_count;
    for (int j = 0; j < nj; ++j)
        if (lbs_->joint_parents[j] >= j)
        {
            fprintf(stderr, "[ARFWriter] --ground: joint %d is listed before its parent, skipped\n", j);
            return;
        }

    // Foot vertices: at least half their skin weight on a foot joint.
    std::vector<char> is_foot(nj, 0);
    for (int j = 0; j < nj && j < mhr_joint_table::N_JOINTS; ++j)
    {
        const char* n = mhr_joint_table::NAMES[j];
        if ((n[0] == 'l' || n[0] == 'r') && n[1] == '_' &&
            (strstr(n, "foot") || strstr(n, "talocrural") || strstr(n, "subtalar") ||
             strstr(n, "transversetarsal") || strstr(n, "ball")))
            is_foot[j] = 1;
    }
    std::vector<float> foot_w(nv, 0.f), side_best(nv, 0.f);
    std::vector<char>  side(nv, 0);   // 0 = left, 1 = right: heaviest foot joint
    for (int i = 0; i < lbs_->n_skin; ++i)
    {
        const int j = lbs_->skin_joint_idx[i], v = lbs_->skin_vert_idx[i];
        if (!is_foot[j]) continue;
        foot_w[v] += lbs_->skin_weights[i];
        if (lbs_->skin_weights[i] > side_best[v])
            { side_best[v] = lbs_->skin_weights[i]; side[v] = mhr_joint_table::NAMES[j][0] == 'r'; }
    }
    std::vector<int> slot(nv, -1), feet;
    for (int v = 0; v < nv; ++v)
        if (foot_w[v] >= 0.5f) { slot[v] = (int)feet.size(); feet.push_back(v); }
    if (feet.empty()) { fprintf(stderr, "[ARFWriter] --ground: no foot vertices, skipped\n"); return; }
    std::vector<std::vector<std::pair<int, float>>> infl(feet.size());
    for (int i = 0; i < lbs_->n_skin; ++i)
    {
        const int s = slot[lbs_->skin_vert_idx[i]];
        if (s >= 0) infl[s].push_back({ lbs_->skin_joint_idx[i], lbs_->skin_weights[i] });
    }

    std::vector<float> ibm((size_t)nj * 16);
    for (int j = 0; j < nj; ++j)
    {
        const float* ib = lbs_->inv_bind_pose + (size_t)j * 8;
        compose_trs_mat4(ib, ib + 3, ib[7], &ibm[(size_t)j*16]);
    }

    // Lowest skinned foot point per frame (cm, Y up), overall and per foot.
    std::vector<std::array<float, 3>> low(nf), foot_low[2];
    foot_low[0].resize(nf); foot_low[1].resize(nf);
    std::vector<float> glob((size_t)nj * 16), skin((size_t)nj * 16);
    for (int f = 0; f < nf; ++f)
    {
        const float* L = &p.joint_mats[(size_t)f * nj * 16];
        for (int j = 0; j < nj; ++j)
        {
            const int par = lbs_->joint_parents[j];
            if (par < 0) std::copy(L + j*16, L + j*16 + 16, &glob[(size_t)j*16]);
            else         mul4(&glob[(size_t)par*16], L + j*16, &glob[(size_t)j*16]);
            mul4(&glob[(size_t)j*16], &ibm[(size_t)j*16], &skin[(size_t)j*16]);
        }
        low[f] = { 0.f, 1e30f, 0.f };
        foot_low[0][f] = foot_low[1][f] = { 0.f, 1e30f, 0.f };
        for (size_t k = 0; k < feet.size(); ++k)
        {
            const float* v = &rest_verts[(size_t)feet[k] * 3];
            float x[3] = { 0.f, 0.f, 0.f };
            for (const auto& jw : infl[k])
            {
                const float* S = &skin[(size_t)jw.first * 16];
                for (int c = 0; c < 3; ++c)
                    x[c] += jw.second * (S[c*4]*v[0] + S[c*4+1]*v[1] + S[c*4+2]*v[2] + S[c*4+3]);
            }
            if (x[1] < low[f][1]) low[f] = { x[0], x[1], x[2] };
            auto& fl = foot_low[(int)side[feet[k]]][f];
            if (x[1] < fl[1]) fl = { x[0], x[1], x[2] };
        }
    }

    // Robust fit of y = c + g . (xz - m) over the frames with a foot down.
    std::vector<char> use(nf, 1);
    float mx = 0.f, mz = 0.f, c0 = 0.f, gx = 0.f, gz = 0.f;
    int   n_axes = 0;
    std::vector<float> res(nf);
    for (int it = 0; it <= GROUND_ITERS; ++it)
    {
        double sx = 0, sz = 0, sy = 0; int n = 0;
        for (int f = 0; f < nf; ++f) if (use[f]) { sx += low[f][0]; sz += low[f][2]; sy += low[f][1]; ++n; }
        mx = (float)(sx / n); mz = (float)(sz / n); c0 = (float)(sy / n);
        double cxx = 0, cxz = 0, czz = 0;
        for (int f = 0; f < nf; ++f) if (use[f])
        {
            const double dx = low[f][0] - mx, dz = low[f][2] - mz;
            cxx += dx*dx; cxz += dx*dz; czz += dz*dz;
        }
        cxx /= n; cxz /= n; czz /= n;
        // Principal horizontal axes of the feet's travel (2x2 symmetric eigen).
        const double tr = cxx + czz, det = cxx*czz - cxz*cxz;
        const double disc = std::sqrt(std::max(0.0, tr*tr/4 - det));
        const double lam[2] = { tr/2 + disc, tr/2 - disc };
        gx = gz = 0.f; n_axes = 0;
        for (int a = 0; a < 2; ++a)
        {
            if (std::sqrt(std::max(0.0, lam[a])) < GROUND_MIN_TRAVEL_CM) continue;
            double ex = cxz, ez = lam[a] - cxx;            // (A - lam I) e = 0
            if (std::fabs(ex) + std::fabs(ez) < 1e-9) { ex = (cxx >= czz) == (a == 0); ez = !ex; }
            const double en = std::sqrt(ex*ex + ez*ez); ex /= en; ez /= en;
            double suy = 0, suu = 0;
            for (int f = 0; f < nf; ++f) if (use[f])
            {
                const double u = (low[f][0] - mx) * ex + (low[f][2] - mz) * ez;
                suy += u * (low[f][1] - c0); suu += u * u;
            }
            const double slope = suy / suu;
            gx += (float)(slope * ex); gz += (float)(slope * ez);
            ++n_axes;
        }
        for (int f = 0; f < nf; ++f)
            res[f] = low[f][1] - (c0 + gx * (low[f][0] - mx) + gz * (low[f][2] - mz));
        if (it == GROUND_ITERS) break;
        std::vector<float> sorted(res);
        const size_t k = (size_t)(GROUND_KEEP * (nf - 1));
        std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end());
        for (int f = 0; f < nf; ++f) use[f] = res[f] <= sorted[k];
    }

    // Plane normal; rotate it onto +Y about the middle of the feet's path.
    float nrm[3] = { -gx, 1.f, -gz };
    const float nl = std::sqrt(nrm[0]*nrm[0] + 1.f + nrm[2]*nrm[2]);
    for (float& c : nrm) c /= nl;
    float tilt = std::acos(std::min(1.f, nrm[1])) * 180.f / 3.14159265f;
    const bool skip_tilt = tilt > GROUND_MAX_TILT_DEG;
    float R[9] = { 1,0,0, 0,1,0, 0,0,1 };
    if (!skip_tilt && tilt > 0.f)
    {
        // Rodrigues: axis = n x Y = (-n.z, 0, n.x), cos = n.y.
        float ax = -nrm[2], az = nrm[0];
        const float s = std::sqrt(ax*ax + az*az), c = nrm[1];
        ax /= s; az /= s;
        const float t = 1.f - c;
        R[0] = t*ax*ax + c; R[1] = -s*az;  R[2] = t*ax*az;
        R[3] = s*az;        R[4] = c;      R[5] = -s*ax;
        R[6] = t*ax*az;     R[7] = s*ax;   R[8] = t*az*az + c;
    }
    // v' = R (v - q) + (mx, 0, mz), q = the plane point above the path's middle.
    const float q[3] = { mx, c0, mz };
    float G[16] = { R[0], R[1], R[2], 0.f,  R[3], R[4], R[5], 0.f,
                    R[6], R[7], R[8], 0.f,  0.f,  0.f,  0.f,  1.f };
    const float shift[3] = { mx, 0.f, mz };
    for (int r = 0; r < 3; ++r)
        G[r*4+3] = shift[r] - (R[r*3]*q[0] + R[r*3+1]*q[1] + R[r*3+2]*q[2]);
    if (skip_tilt)   // level floor only: drop onto the median contact height
    {
        std::vector<float> ys; for (int f = 0; f < nf; ++f) if (use[f]) ys.push_back(low[f][1]);
        G[7] = -median_of(ys);
    }

    for (int f = 0; f < nf; ++f)
        for (int j = 0; j < nj; ++j)
            if (lbs_->joint_parents[j] < 0)
            {
                float* M = &p.joint_mats[((size_t)f * nj + j) * 16];
                float out[16]; mul4(G, M, out);
                std::copy(out, out + 16, M);
            }

    // ── Per frame: the planted foot onto the floor ─────────────────────────
    // The feet in the grounded space: p' = G p.
    for (auto& fl : foot_low)
        for (auto& x : fl)
        {
            const std::array<float, 3> o = x;
            for (int r = 0; r < 3; ++r)
                x[r] = G[r*4]*o[0] + G[r*4+1]*o[1] + G[r*4+2]*o[2] + G[r*4+3];
        }
    std::vector<float> dy(nf, 0.f);
    std::vector<char>  planted(nf, 0);
    const float fps = 1.f / frame_time_;
    for (int f = 0; f < nf; ++f)
    {
        const int f0 = std::max(0, f - GROUND_SPEED_HALF), f1 = std::min(nf - 1, f + GROUND_SPEED_HALF);
        for (const auto& fl : foot_low)
        {
            const float dx = fl[f1][0] - fl[f0][0], dz = fl[f1][2] - fl[f0][2];
            const float speed = f1 > f0 ? std::sqrt(dx*dx + dz*dz) * fps / (float)(f1 - f0) : 0.f;
            if (speed < GROUND_PLANT_CMS) planted[f] = 1;
        }
        // The lower foot, planted or sliding, is the one on the floor.
        if (planted[f]) dy[f] = -std::min(foot_low[0][f][1], foot_low[1][f][1]);
    }
    int n_planted = 0, prev = -1;
    for (int f = 0; f < nf; ++f)
    {
        if (!planted[f]) continue;
        ++n_planted;
        if (prev < 0)            for (int g = 0; g < f; ++g) dy[g] = dy[f];   // hold at the start
        else for (int g = prev + 1; g < f; ++g)
            dy[g] = dy[prev] + (dy[f] - dy[prev]) * (float)(g - prev) / (float)(f - prev);
        prev = f;
    }
    if (prev < 0) std::fill(dy.begin(), dy.end(), 0.f);         // never planted: rigid only
    else for (int g = prev + 1; g < nf; ++g) dy[g] = dy[prev];  // hold at the end
    // Snapping to each frame's foot height and switching between snapped and
    // interpolated frames makes the pelvis jerk; two passes of a centred box
    // (+-GROUND_SMOOTH_HALF frames, ~triangular) take that out.  Then no
    // frame may push a foot below Y = 0 (smoothing, or a fast-moving foot
    // that still brushes the floor in a dance step).
    for (int pass = 0; pass < 2; ++pass)
    {
        const std::vector<float> in(dy);
        for (int f = 0; f < nf; ++f)
        {
            const int a = std::max(0, f - GROUND_SMOOTH_HALF), b = std::min(nf - 1, f + GROUND_SMOOTH_HALF);
            float sum = 0.f;
            for (int g = a; g <= b; ++g) sum += in[g];
            dy[f] = sum / (float)(b - a + 1);
        }
    }
    for (int f = 0; f < nf; ++f)
        dy[f] = std::max(dy[f], -std::min(foot_low[0][f][1], foot_low[1][f][1]));

    for (int f = 0; f < nf; ++f)
        for (int j = 0; j < nj; ++j)
            if (lbs_->joint_parents[j] < 0) p.joint_mats[((size_t)f * nj + j) * 16 + 7] += dy[f];

    std::vector<float> shifts(dy.begin(), dy.end());
    for (float& v : shifts) v = std::fabs(v);
    std::sort(shifts.begin(), shifts.end());
    const float p90 = shifts[(size_t)(0.9f * (nf - 1))];
    if (skip_tilt)
        printf("[ARFWriter] person %d: --ground fitted a %.1f deg floor, implausible, levelled only\n", p.id, tilt);
    else if (n_axes == 0)
        printf("[ARFWriter] person %d: --ground: feet travel under %.0f cm, floor tilt unmeasurable, "
               "dropped onto the floor only\n", p.id, GROUND_MIN_TRAVEL_CM);
    else
        printf("[ARFWriter] person %d: --ground floor tilted %.1f deg (%s)\n", p.id, tilt,
               n_axes == 2 ? "both axes" : "along the feet's travel");
    printf("[ARFWriter] person %d: --ground planted a foot in %d of %d frames; per-frame height fix "
           "under %.1f cm in 90%% of frames\n", p.id, n_planted, nf, p90);
}

bool ARFWriter::dump_one_person(PerPerson& p)
{
    if (p.frame_count == 0) return true;   // nothing tracked long enough to write

    const int nj = lbs_->n_joints;
    const int nv = lbs_->n_verts;

    // ── Personalized rest mesh: base_shape + averaged identity-shape blend ──
    std::vector<double> shape_avg(lbs_->n_shape_pc, 0.0);
    for (int k = 0; k < lbs_->n_shape_pc && k < (int)p.shape_sum.size(); ++k)
        shape_avg[k] = p.shape_sum[k] / std::max(1, p.shape_samples);

    std::vector<float> verts((size_t)nv * 3);
    for (size_t i = 0; i < verts.size(); ++i)
    {
        double acc = lbs_->base_shape[i];
        for (int k = 0; k < lbs_->n_shape_pc; ++k)
            acc += shape_avg[k] * (double)lbs_->shape_vectors[(size_t)k * nv * 3 + i];
        verts[i] = (float)acc;
    }

    // ── Rest skeleton offsets: median measured bone vector per joint (this
    // person's actual proportions) instead of the raw template offsets —
    // mirrors BVHWriter::rewrite_offsets_for. Root keeps its raw offset (no
    // parent to measure a bone vector against).
    std::vector<std::array<float,3>> rest_offset(nj);
    for (int j = 0; j < nj; ++j)
    {
        if (lbs_->joint_parents[j] < 0 || p.bone_samples[j].size() < 3)
        {
            rest_offset[j] = { lbs_->joint_offsets[j*3+0],
                               lbs_->joint_offsets[j*3+1],
                               lbs_->joint_offsets[j*3+2] };
        }
        else
        {
            float m[3]; median_vec3(p.bone_samples[j], m);
            rest_offset[j] = { m[0], m[1], m[2] };
        }
    }

    if (ground_) ground_person(p, verts);

    // ── Binary data items ───────────────────────────────────────────────────
    // data[].id is a running count assigned in this fixed order: mesh_positions=0,
    // mesh_indices=1, skin_weights=2, inverse_bind=3, then one id per blendshape
    // shape (if export_face_) — mirrors ARFPlayer's arfPlanDataIds() convention
    // (see doc/CONFORMANCE_GAPS.md "For SAM3DBody-cpp").
    struct DataItem { int id; std::string name, path, mime; Bytes bytes; };
    std::vector<DataItem> items;

    items.push_back({ 0, "mesh_positions",
        "data/mesh_positions.bin", "application/mpeg.arf.dense",
        build_dense_tensor({ (int32_t)nv, 3 }, GLTF_FLOAT, verts.data(), verts.size()*sizeof(float)) });

    const unsigned int ni = mesh_->header.numberOfIndices;
    std::vector<uint32_t> idx(mesh_->indices, mesh_->indices + ni);
    const uint32_t n_tris = ni / 3;
    items.push_back({ 1, "mesh_indices",
        "data/mesh_indices.bin", "application/mpeg.arf.dense",
        build_dense_tensor({ (int32_t)n_tris, 3 }, GLTF_UNSIGNED_INT,
                           idx.data(), idx.size()*sizeof(uint32_t)) });

    {
        std::vector<uint32_t> flat_idx(lbs_->n_skin);
        std::vector<float>    weights(lbs_->skin_weights, lbs_->skin_weights + lbs_->n_skin);
        for (int i = 0; i < lbs_->n_skin; ++i)
            flat_idx[i] = (uint32_t)lbs_->skin_vert_idx[i] * (uint32_t)nj
                        + (uint32_t)lbs_->skin_joint_idx[i];
        items.push_back({ 2, "skin_weights",
            "data/skin_weights.bin", "application/mpeg.arf.sparse",
            build_sparse_tensor({ (int32_t)nv, (int32_t)nj }, flat_idx, weights) });
    }

    {
        std::vector<float> ibm((size_t)nj * 16);
        for (int j = 0; j < nj; ++j)
        {
            const float* ib = lbs_->inv_bind_pose + (size_t)j * 8;  // tx,ty,tz,qx,qy,qz,qw,scale
            compose_trs_mat4(ib, ib + 3, ib[7], &ibm[(size_t)j*16]);
        }
        items.push_back({ 3, "inverse_bind_matrices",
            "data/inv_bind_pose.bin", "application/mpeg.arf.dense",
            build_dense_tensor({ (int32_t)nj, 16 }, GLTF_FLOAT, ibm.data(), ibm.size()*sizeof(float)) });
    }

    // BlendshapeSet.shapes: one minimal GLB per face PCA basis vector, storing
    // its ABSOLUTE deformed positions (base mesh + delta), per spec — see the
    // BlendshapeSet.shapes note in ARF.md. Each GLB's topology (indices) is
    // byte-identical to the base mesh.
    const int n_face_shapes = export_face_ ? lbs_->n_face_pc : 0;
    for (int s = 0; s < n_face_shapes; ++s)
    {
        std::vector<float> shape_pos((size_t)nv * 3);
        const float* delta = lbs_->face_vectors + (size_t)s * nv * 3;
        for (size_t i = 0; i < shape_pos.size(); ++i)
            shape_pos[i] = verts[i] + delta[i];

        char name[48], path[48];
        std::snprintf(name, sizeof(name), "face_blendshape_%d", s);
        std::snprintf(path, sizeof(path), "data/face_blendshape_%d.glb", s);
        items.push_back({ 4 + s, name, path, "model/gltf-binary",
            build_glb_mesh(shape_pos, (uint32_t)nv, idx, n_tris) });
    }

    // TextureSet (--skin-color): the mesh has no UVs, so the appearance travels
    // as a GLB of this rest mesh with per-vertex COLOR_0 — the "GLB material"
    // case libarf's materialPath note allows.  libarf rejects a TextureSet
    // without targets, so its one target names the same item (ARF.md).
    const auto colors_it = colors_.find(p.id);
    const bool has_colors = colors_it != colors_.end() &&
                            colors_it->second.size() == (size_t)nv * 3;
    const int  texture_id = 4 + n_face_shapes;
    if (has_colors)
        items.push_back({ texture_id, "skin_color", "data/skin_color.glb", "model/gltf-binary",
            build_glb_mesh(verts, (uint32_t)nv, idx, n_tris, &colors_it->second) });

    // ── Animation streams ───────────────────────────────────────────────────
    // joint_set_id/blendshape_set_id are the declared ids of skeletons[0]/
    // blendshapeSets[0] — always 0, this writer's own convention (see
    // components below).
    const float timescale = 1.0f / frame_time_;   // ticks/sec == fps (1 tick = 1 frame)
    Bytes joint_stream;
    append_aau(joint_stream, AAU_CONFIG, build_config_payload("arf-body-v1", timescale));
    for (int f = 0; f < p.frame_count; ++f)
        append_aau(joint_stream, AAU_JOINT,
                  build_joint_payload((uint32_t)f, 0, &p.joint_mats[(size_t)f*nj*16], nj));

    Bytes face_stream;
    if (export_face_)
    {
        const int nfp = lbs_->n_face_pc;
        append_aau(face_stream, AAU_CONFIG, build_config_payload("arf-face-v1", timescale));
        for (int f = 0; f < p.frame_count; ++f)
            append_aau(face_stream, AAU_BLENDSHAPE,
                      build_blendshape_payload((uint32_t)f, 0,
                                               &p.face_weights[(size_t)f*nfp], nfp));
    }

    // ── arf.json ─────────────────────────────────────────────────────────────
    using arf_json::Value;

    Value nodes = Value::array();
    Value joints_arr = Value::array();
    for (int j = 0; j < nj; ++j)
    {
        const char* name = (j < mhr_joint_table::N_JOINTS) ? mhr_joint_table::NAMES[j] : "?";
        Value node = Value::object();
        node.set("id", j);
        node.set("name", name);
        // mapping is a mandatory semantic scene-graph path in the spec (see
        // its companion scene-description part, 23090-14); this project has
        // no verified taxonomy for it, so the node's own name stands in as an
        // honest single-segment placeholder — see ARF.md.
        node.set("mapping", name);
        if (lbs_->joint_parents[j] >= 0)
            node.set("parent", lbs_->joint_parents[j]);
        node.set("translation", Value::array()
                    .push_back(rest_offset[j][0]).push_back(rest_offset[j][1]).push_back(rest_offset[j][2]));
        const float* q = lbs_->joint_prerotations + j*4;
        node.set("rotation", Value::array()
                    .push_back(q[0]).push_back(q[1]).push_back(q[2]).push_back(q[3]));
        nodes.push_back(node);
        joints_arr.push_back(j);
    }

    Value skeleton = Value::object();
    skeleton.set("id", 0);
    skeleton.set("name", "skeleton0");
    skeleton.set("root", 0);   // node id 0 == mhr_joint_table::NAMES[0] == "body_world"
    skeleton.set("joints", joints_arr);
    skeleton.set("inverseBindMatrix", 3);   // data[] id

    Value skin = Value::object();
    skin.set("id", 0);
    skin.set("name", "skin0");
    skin.set("mapping", "skin0");
    skin.set("skeleton", 0);
    skin.set("mesh", 0);
    skin.set("weights", 2);   // data[] id
    if (has_colors) skin.set("textureSet", 0);   // the TextureSet's only link

    Value mesh = Value::object();
    mesh.set("id", 0);
    mesh.set("name", "mesh0");
    mesh.set("path", "mesh0");
    mesh.set("data", Value::array().push_back(0).push_back(1));   // [positions id, indices id]

    Value components = Value::object();
    components.set("nodes", nodes);
    components.set("skeletons", Value::array().push_back(skeleton));
    components.set("skins", Value::array().push_back(skin));
    components.set("meshes", Value::array().push_back(mesh));

    if (export_face_)
    {
        Value shapes = Value::array();
        for (int s = 0; s < n_face_shapes; ++s) shapes.push_back(4 + s);

        Value bs = Value::object();
        bs.set("id", 0);
        bs.set("name", "face_expression");
        bs.set("baseMesh", 0);
        bs.set("shapes", shapes);
        components.set("blendshapeSets", Value::array().push_back(bs));
    }

    if (has_colors)
    {
        Value target = Value::object();
        target.set("id", 0);
        target.set("name", "skin_color");
        target.set("texture", texture_id);
        target.set("texturePath", "");
        Value ts = Value::object();
        ts.set("id", 0);
        ts.set("name", "skin_color");
        ts.set("animationInfo", Value::array());
        ts.set("material", texture_id);
        ts.set("materialPath", "");
        ts.set("targets", Value::array().push_back(target));
        components.set("textureSets", Value::array().push_back(ts));
    }

    Value preamble = Value::object();
    preamble.set("signature", "ARF");
    preamble.set("version", "1.0");
    // supportedAnimations is a SupportedAnimations object (bodyAnimations/
    // faceAnimations/..., each an array of profile strings), not a flat
    // array of profile-name strings — matches libarf's arfWriteJson(), see
    // ARFPlayer's doc/CONFORMANCE_GAPS.md "Preamble / Metadata".
    Value supported = Value::object();
    supported.set("bodyAnimations", Value::array().push_back("arf-body-v1"));
    if (export_face_) supported.set("faceAnimations", Value::array().push_back("arf-face-v1"));
    preamble.set("supportedAnimations", supported);

    Value metadata = Value::object();
    // "SAM3DBody-cpp", not "SAM3DBody": ARF export exists only in this C++
    // port, so naming upstream here would credit a producer that cannot emit
    // these containers.
    metadata.set("name", "SAM3DBody-cpp avatar");
    metadata.set("id", std::string("person_") + std::to_string(p.id));
    // age/gender are mandatory per the Metadata schema; this pipeline has no
    // age/gender source, so these are honest placeholders (matching libarf's
    // own arfCreate() defaults), not fabricated data.
    metadata.set("age", -1);
    metadata.set("gender", std::string("unspecified"));

    Value lod = Value::object();
    lod.set("name", "lod0");
    lod.set("skins", Value::array().push_back(0));
    lod.set("meshes", Value::array().push_back(0));
    lod.set("skeletons", Value::array().push_back(0));
    if (export_face_) lod.set("blendshapeSets", Value::array().push_back(0));
    if (has_colors)   lod.set("textureSets", Value::array().push_back(0));

    Value asset = Value::object();
    asset.set("name", "body");
    asset.set("isMain", true);
    asset.set("lods", Value::array().push_back(lod));

    Value data = Value::array();
    for (const auto& it : items)
        data.push_back(Value::object()
            .set("id", it.id).set("name", it.name).set("uri", it.path)
            .set("type", it.mime).set("byteLength", (int)it.bytes.size()));

    Value doc = Value::object();
    doc.set("preamble", preamble);
    doc.set("metadata", metadata);
    // No field names the animation streams' location: animations/joints.bin
    // and animations/face.bin are found by fixed path per the Zip-container
    // clause — see ARF.md "Avatar Animation Units".
    doc.set("structure", Value::object().set("assets", Value::array().push_back(asset)));
    doc.set("components", components);
    doc.set("data", data);

    const std::string json_text = doc.dump(2);

    // ── id_map.txt (non-normative debug sidecar; see ARF.md) ───────────────
    std::string id_map;
    for (int j = 0; j < nj; ++j)
    {
        const char* name = (j < mhr_joint_table::N_JOINTS) ? mhr_joint_table::NAMES[j] : "?";
        id_map += "node\t" + std::to_string(j) + "\t" + name + "\n";
    }
    id_map += "mesh\t0\tmesh0\n";
    id_map += "skin\t0\tskin0\n";
    id_map += "skeleton\t0\tskeleton0\n";
    if (export_face_) id_map += "blendshapeSet\t0\tface_expression\n";
    if (has_colors)   id_map += "textureSet\t0\tskin_color\n";

    // ── Assemble the .arfz ZIP container ────────────────────────────────────
    const std::string out_file = per_person_path(out_path_, id_prefix_, p.id);
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    if (!mz_zip_writer_init_file(&zip, out_file.c_str(), 0))
    {
        fprintf(stderr, "[ARFWriter] cannot open '%s' for writing\n", out_file.c_str());
        return false;
    }

    bool ok = mz_zip_writer_add_mem(&zip, "arf.json", json_text.data(), json_text.size(),
                                    MZ_BEST_SPEED);
    ok = ok && mz_zip_writer_add_mem(&zip, "id_map.txt", id_map.data(), id_map.size(),
                                     MZ_BEST_SPEED);
    for (const auto& it : items)
        ok = ok && mz_zip_writer_add_mem(&zip, it.path.c_str(), it.bytes.data(), it.bytes.size(),
                                         MZ_BEST_SPEED);
    ok = ok && mz_zip_writer_add_mem(&zip, "animations/joints.bin",
                                     joint_stream.data(), joint_stream.size(), MZ_BEST_SPEED);
    if (export_face_)
        ok = ok && mz_zip_writer_add_mem(&zip, "animations/face.bin",
                                         face_stream.data(), face_stream.size(), MZ_BEST_SPEED);

    ok = ok && mz_zip_writer_finalize_archive(&zip);
    mz_zip_writer_end(&zip);

    if (!ok)
    {
        fprintf(stderr, "[ARFWriter] failed writing '%s'\n", out_file.c_str());
        return false;
    }
    fprintf(stderr, "[ARFWriter] wrote %s (%d frames)\n", out_file.c_str(), p.frame_count);
    return true;
}

void ARFWriter::close()
{
    if (!lbs_) return;
    for (auto& kv : people_) dump_one_person(kv.second);

    mhr_lbs_free(lbs_);
    lbs_ = nullptr;
    if (mesh_) { tri_freeModel(mesh_); mesh_ = nullptr; }
    tracks_.clear();
    people_.clear();
    colors_.clear();
}
