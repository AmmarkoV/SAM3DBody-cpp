#pragma once
// RobustCapture: a cv::VideoCapture-compatible wrapper that only changes
// behavior for real V4L2 webcam devices (/dev/video* or a numeric camera
// index). Video files and images are untouched -- they fall through to a
// plain cv::VideoCapture, either because the source clearly isn't a V4L2
// device or because the raw V4L2 open/negotiation failed for any reason.
//
// Why this exists: cv::VideoCapture's V4L2 backend does not check the
// V4L2_BUF_FLAG_ERROR flag the kernel sets on a dequeued buffer when a USB
// isochronous transfer for that frame was incomplete (packet loss under
// bus contention is routine on UVC webcams). It hands back that buffer's
// stale/partial contents as if it were a normal frame -- visible as an
// occasional torn/half-updated image, independent of the GPU or vsync
// behavior anywhere downstream. RobustCapture re-dequeues instead of
// handing back a buffer the kernel itself flagged as bad.
//
// Only the subset of the cv::VideoCapture API actually used by the live
// renderers is implemented: open/isOpened, set/get for FOURCC, FRAME_WIDTH,
// FRAME_HEIGHT, FPS, BUFFERSIZE, grab/retrieve/read and operator>>.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <memory>
#include <string>

class RobustCapture {
public:
    RobustCapture();
    ~RobustCapture();

    bool open(int index);
    bool open(const std::string& path);
    bool isOpened() const;

    bool set(int prop, double value);
    double get(int prop) const;

    bool grab();
    bool retrieve(cv::Mat& frame);
    bool read(cv::Mat& frame);
    RobustCapture& operator>>(cv::Mat& frame);

    void release();

private:
    struct V4L2Impl;
    std::unique_ptr<V4L2Impl> v4l2_;
    cv::VideoCapture cv_cap_;
    bool use_v4l2_ = false;

    bool openV4L2(const std::string& devicePath);
};
