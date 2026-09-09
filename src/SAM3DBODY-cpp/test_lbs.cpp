#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include "fast_sam_3dbody_capi.h"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <onnx_dir> <image>\n", argv[0]);
        return 1;
    }

    std::string onnx_dir = argv[1];
    std::string image    = argv[2];
    std::string gguf_path = onnx_dir + "/pipeline.gguf";
    std::string yolo_path = onnx_dir + "/yolo.onnx";

    // Load image
    cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Cannot read image: %s\n", image.c_str());
        return 1;
    }
    printf("Image: %dx%d\n", img.cols, img.rows);

    // Create pipeline
    FsbHandle h = fsb_create();
    if (!h) { fprintf(stderr, "fsb_create failed\n"); return 1; }

    FsbConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.onnx_dir        = onnx_dir.c_str();
    cfg.gguf_path       = gguf_path.c_str();
    cfg.yolo_path       = yolo_path.c_str();
    cfg.cuda_device     = 0;
    cfg.skip_body_model = 0;   /* Native C LBS */
    cfg.person_thresh   = 0.5f;
    cfg.person_nms_iou  = 0.45f;
    cfg.max_persons     = 0;
    cfg.focal_x         = 0.f;
    cfg.focal_y         = 0.f;
    cfg.principal_x     = 0.f;
    cfg.principal_y     = 0.f;

    printf("Loading...\n");
    if (!fsb_load(h, &cfg)) {
        fprintf(stderr, "fsb_load failed\n");
        fsb_destroy(h);
        return 1;
    }

    // Process
    printf("Processing...\n");
    FsbResult results[32];
    memset(results, 0, sizeof(results));

    fflush(stdout);
    int n = fsb_process_bgr(h, img.data, img.cols, img.rows, results, 32);
    printf("Done! Got %d results\n", n);

    for (int i = 0; i < n; ++i) {
        printf("  Person %d: bbox=[%.0f,%.0f,%.0f,%.0f] has_kps=%d has_yolo=%d\n",
               i, results[i].bbox[0], results[i].bbox[1],
               results[i].bbox[2], results[i].bbox[3],
               results[i].has_kps, results[i].has_yolo_kps);
    }

    fsb_destroy(h);
    return 0;
}
