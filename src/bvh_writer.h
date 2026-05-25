#pragma once
// bvh_writer.h - BVH export for MHR body-pose inference output.
//
// Usage:
//   BVHWriter w;
//   if (!w.open("body.bvh", "output.bvh", 1.f/30.f, "./onnx/body_model.lbs"))
//       { /* error */ }
//   for (auto& frame_results : ...) w.write_frame(frame_results);
//   w.close();   // or let the destructor handle it
//
// All 23 major body joints (arms, collar, head, spine, legs) are animated.
// Note: the MHR model's "hand PCA" encodes arm/collar/head joints, NOT fingers,
// so finger channels in the BVH remain at zero (T-pose hands).

#include <string>
#include <vector>

namespace fsb { struct MHRResult; }

class BVHWriter
{
public:
    // Load hierarchy from template_path, open out_path for writing.
    // lbs_path: unused (kept for call-site compatibility).
    // Returns false (and prints to stderr) on any error.
    bool open(const std::string& template_path,
              const std::string& out_path,
              float frame_time = 0.04f,
              const std::string& lbs_path = "");

    // Append one frame (person[0] used when available; zero frame otherwise).
    void write_frame(const std::vector<fsb::MHRResult>& results);

    // Patch the "Frames:" count and close the file.
    void close();

    bool is_open()          const { return file_ != nullptr; }
    int  frame_count()      const { return frame_count_;     }
    int  channels_per_frame()const { return total_channels_; }

    BVHWriter()  = default;
    ~BVHWriter() { if (is_open()) close(); }
    BVHWriter(const BVHWriter&)            = delete;
    BVHWriter& operator=(const BVHWriter&) = delete;

private:
    FILE*  file_           = nullptr;
    long   frames_pos_     = 0;
    int    frame_count_    = 0;
    int    total_channels_ = 0;
    float  frame_time_     = 0.04f;

    // Major body joints (22 active + 1 skipped).
    struct ResolvedJoint {
        int bvh_offset;   // offset in the 498-channel frame row (-1 = skip)
        int rx_idx;       // index into body_pose[] for Xrotation
        int ry_idx;       // index for Yrotation
        int rz_idx;       // index for Zrotation
    };
    ResolvedJoint resolved_[23]{};

    void build_frame(const fsb::MHRResult& r, float* buf) const;
};
