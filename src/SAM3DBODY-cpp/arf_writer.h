#pragma once
// arf_writer.h — MPEG ARF (ISO/IEC 23090-39) export for the MHR body-pose
// pipeline.
//
// ARF ("Avatar Representation Format") is a JSON-described base-avatar model
// (skeleton + skins + meshes + blendshape sets) packaged in a container, plus
// a binary Animation Stream of timestamped Avatar Animation Units (AAUs).
// See ARF.md for exactly which subset this writer implements and where it
// deviates from the spec's own examples (e.g. raw dense-tensor mesh/skin
// data instead of embedded glTF/GLB). Spec overview cited there:
//   J. Regateiro, A. Trioux, Q. Avril, "The MPEG Avatar Representation
//   Format (ARF): An Interoperable Container and Animation Framework for
//   Avatars," IEEE Comp. Graph. & App., 2026,
//   https://ieeexplore.ieee.org/document/11667221
//
// Mirrors BVHWriter's lifecycle (open/write_frame[_external]/close) and
// reuses its shared MHR FK core (mhr_fk.h) so both exporters run identical
// kinematics on the same MHRResult stream. One .arfz ZIP container is
// written per tracked person: "<out_path>_<id>.arfz".

#include <string>
#include <unordered_map>
#include <vector>

#include "mhr_fk.h"

struct MHR_LBS_Data;
struct TRI_Model;
namespace fsb { struct MHRResult; }

class ARFWriter
{
public:
    // lbs_path:  <onnx_dir>/body_model.lbs — skeleton, skin weights,
    //            identity-shape and face-blendshape bases.
    // mesh_path: <onnx_dir>/body_mesh.tri — triangle topology only; vertex
    //            POSITIONS come from lbs_path's base_shape (same convention
    //            fast_sam_3dbody_render's write_obj_mesh() already uses).
    // export_face: also write the facial BlendshapeSet + its AAU_BLENDSHAPE
    //            animation track. Only meaningful when the pipeline ran with
    //            --dev-face (PipelineConfig::zero_face_params == false) —
    //            otherwise MHRResult::face_params is always zero and this
    //            would just bloat the output with a static "blendshape".
    bool open(const std::string& out_path,
              const std::string& lbs_path,
              const std::string& mesh_path,
              float              frame_time  = 1.0f / 30.0f,
              bool               export_face = false);

    // Offline path: caller supplies track ids (from the global tracker).
    // Same contract as BVHWriter::write_frame_external.
    void write_frame_external(const std::vector<fsb::MHRResult>& results,
                              const std::vector<int>& track_ids,
                              const std::vector<int>& pad_ids = {});

    // Live path: internal greedy-IoU tracker. Same contract as
    // BVHWriter::write_frame.
    void write_frame(const std::vector<fsb::MHRResult>& results);

    // Optional label inserted before the numeric id in per-person filenames:
    // "<stem>_<prefix><id>.arfz" — mirrors BVHWriter::set_id_label_prefix,
    // shared by the --bvh-split-scenes path for ARF output (ARF.md).
    void set_id_label_prefix(const std::string& p) { id_prefix_ = p; }

    void close();
    bool is_open() const { return lbs_ != nullptr; }

    ARFWriter()  = default;
    ~ARFWriter() { if (is_open()) close(); }
    ARFWriter(const ARFWriter&)            = delete;
    ARFWriter& operator=(const ARFWriter&) = delete;

private:
    struct PerPerson
    {
        int id          = -1;
        int frame_count = 0;

        // AAU_JOINT stream: one 4x4 row-major local transform (16 floats)
        // per joint per frame.
        std::vector<float> joint_mats;      // [frame_count * n_joints * 16]

        // AAU_BLENDSHAPE stream (only when export_face_): one weight per
        // face blendshape per frame.
        std::vector<float> face_weights;    // [frame_count * n_face_pc]

        // Identity-shape accumulation, baked into a personalized rest mesh
        // at close() instead of streamed as a live BlendshapeSet — see
        // ARF.md "Identity shape is baked, not streamed".
        std::vector<double> shape_sum;      // [n_shape_pc]
        int                 shape_samples = 0;

        // Per-joint rest-local bone-vector samples (this joint to its DIRECT
        // MHR parent — see mhr_fk::State::rest_local_bone_vector), median-
        // rewritten into the Skeleton's Node offsets at close(). Mirrors
        // BVHWriter::rewrite_offsets_for's measured-bone-length philosophy.
        std::vector<std::vector<float>> bone_samples;  // [n_joints][3*n_samples]
    };

    struct Track { int id; float bbox[4]; int last_seen_frame; };

    MHR_LBS_Data*     lbs_  = nullptr;
    struct TRI_Model* mesh_ = nullptr;   // topology only (indices)
    std::string       out_path_;
    std::string       id_prefix_;
    mhr_fk::State     fk_;
    float             frame_time_     = 1.0f / 30.0f;
    bool              export_face_    = false;
    int               session_frames_ = 0;

    std::vector<Track>                 tracks_;
    int                                 next_track_id_ = 0;
    std::unordered_map<int, PerPerson>  people_;

    std::vector<int> assign_tracks(const std::vector<fsb::MHRResult>& results);
    static float bbox_iou(const float a[4], const float b[4]);

    void append_frame_for(PerPerson& p, const fsb::MHRResult& r);
    void pad_continuation_frame(PerPerson& p);
    bool dump_one_person(const PerPerson& p);
};
