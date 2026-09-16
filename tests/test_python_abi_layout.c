/* Compile against the real public C header; no model/runtime dependency. */
#include "fast_sam_3dbody_capi.h"
#include <stdio.h>

#define FIELD(type, member) \
    printf(",\"" #member "\":[%zu,%zu]", offsetof(type, member), \
           sizeof(((type*)0)->member))

int main(void)
{
    printf("{\"abi_version\":%u,\"FsbConfig\":{\"size\":%zu",
           FSB_ABI_VERSION, sizeof(FsbConfig));
    FIELD(FsbConfig, onnx_dir);
    FIELD(FsbConfig, gguf_path);
    FIELD(FsbConfig, yolo_path);
    FIELD(FsbConfig, cuda_device);
    FIELD(FsbConfig, skip_body_model);
    FIELD(FsbConfig, person_thresh);
    FIELD(FsbConfig, person_nms_iou);
    FIELD(FsbConfig, max_persons);
    FIELD(FsbConfig, focal_x);
    FIELD(FsbConfig, focal_y);
    FIELD(FsbConfig, principal_x);
    FIELD(FsbConfig, principal_y);
    FIELD(FsbConfig, zero_face_params);
    FIELD(FsbConfig, detector);
    printf("},\"FsbResult\":{\"size\":%zu", sizeof(FsbResult));
    FIELD(FsbResult, bbox);
    FIELD(FsbResult, focal_length);
    FIELD(FsbResult, pred_cam_t);
    FIELD(FsbResult, global_rot);
    FIELD(FsbResult, body_pose);
    FIELD(FsbResult, shape);
    FIELD(FsbResult, scale);
    FIELD(FsbResult, hand_pose);
    FIELD(FsbResult, face_params);
    FIELD(FsbResult, yolo_kps);
    FIELD(FsbResult, has_yolo_kps);
    FIELD(FsbResult, kps_3d);
    FIELD(FsbResult, kps_2d);
    FIELD(FsbResult, has_kps);
    FIELD(FsbResult, pred_pose_raw);
    FIELD(FsbResult, pred_cam_raw);
    FIELD(FsbResult, mhr_model_params);
    FIELD(FsbResult, skel_3d);
    FIELD(FsbResult, has_skel);
    printf("}}\n");
    return 0;
}
