// ============================================================================
// fast_sam_3dbody_capi.cpp  –  Plain C wrapper for ctypes / cffi
// ============================================================================

#include "fast_sam_3dbody_capi.h"
#include "fast_sam_3dbody.h"

#include <algorithm>
#include <cstring>

extern "C" {

    FsbHandle fsb_create(void)
    {
        return static_cast<FsbHandle>(new fsb::Pipeline());
    }

    void fsb_destroy(FsbHandle h)
    {
        if (h)
        {
            auto* p = static_cast<fsb::Pipeline*>(h);
            p->free();
            delete p;
        }
    }

    int fsb_load(FsbHandle h, const FsbConfig* cfg)
    {
        if (!h || !cfg) return 0;
        auto* p = static_cast<fsb::Pipeline*>(h);

        fsb::PipelineConfig pc;
        if (cfg->onnx_dir)   pc.onnx_dir      = cfg->onnx_dir;
        if (cfg->gguf_path)  pc.gguf_path     = cfg->gguf_path;
        if (cfg->yolo_path)  pc.yolo_path     = cfg->yolo_path;
        pc.cuda_device      = cfg->cuda_device;
        pc.skip_body_model  = cfg->skip_body_model != 0;
        pc.person_thresh    = cfg->person_thresh;
        pc.person_nms_iou   = cfg->person_nms_iou;
        pc.max_persons      = cfg->max_persons;
        pc.detector         = cfg->detector;
        pc.focal_x          = cfg->focal_x;
        pc.focal_y          = cfg->focal_y;
        pc.principal_x      = cfg->principal_x;
        pc.principal_y      = cfg->principal_y;
        pc.zero_face_params = cfg->zero_face_params != 0;

        return p->load(pc) ? 1 : 0;
    }

    int fsb_process_bgr(FsbHandle      h,
                        const uint8_t* bgr,
                        int            width,
                        int            height,
                        FsbResult*     results,
                        int            max_results)
    {
        if (!h || !bgr || !results || max_results <= 0) return 0;
        auto* p = static_cast<fsb::Pipeline*>(h);

        std::vector<fsb::MHRResult> res = p->process_bgr(bgr, width, height);
        int n = std::min((int)res.size(), max_results);
        printf("[CAPI] process_bgr returned %d results\n", n);

        for (int i = 0; i < n; ++i)
        {
            const fsb::MHRResult& r = res[i];
            FsbResult& out = results[i];
            std::memset(&out, 0, sizeof(FsbResult));

            std::memcpy(out.bbox,        r.bbox.data(),        4  * sizeof(float));
            out.focal_length = r.focal_length;
            std::memcpy(out.pred_cam_t,  r.pred_cam_t.data(),  3  * sizeof(float));
            std::memcpy(out.global_rot,  r.global_rot.data(),  3  * sizeof(float));

            auto copy_vec = [](float* dst, const std::vector<float>& src, int n)
            {
                int cnt = std::min((int)src.size(), n);
                std::memcpy(dst, src.data(), cnt * sizeof(float));
            };

            copy_vec(out.body_pose,   r.body_pose,   133);
            copy_vec(out.shape,       r.shape,        45);
            copy_vec(out.scale,       r.scale,        28);
            copy_vec(out.hand_pose,   r.hand_pose,   108);
            copy_vec(out.face_params, r.face_params,  72);

            if (!r.keypoints_yolo.empty())
            {
                copy_vec(out.yolo_kps, r.keypoints_yolo, 51);
                out.has_yolo_kps = 1;
            }

            if (!r.keypoints_3d.empty())
            {
                copy_vec(out.kps_3d, r.keypoints_3d, 210);
                copy_vec(out.kps_2d, r.keypoints_2d, 140);
                out.has_kps = 1;
            }

            // ── Second-pass raw fields ─────────────────────────────────────────
            // pred_pose_raw[266]: global_rot_6d[6] + body_cont[260], needed by
            // the Python second-pass to reconstruct prev_estimate for forward_decoder.
            std::memcpy(out.pred_pose_raw, r.pred_pose_raw.data(), 266 * sizeof(float));
            // pred_cam_raw[3]: raw cam FFN output before s/tx/ty → pred_cam_t.
            std::memcpy(out.pred_cam_raw,  r.pred_cam_raw.data(),  3   * sizeof(float));

            // mhr_model_params[204]: assembled model_params with hand + scale decoded.
            std::memcpy(out.mhr_model_params, r.mhr_model_params.data(), 204 * sizeof(float));
        }
        return n;
    }

} // extern "C"
