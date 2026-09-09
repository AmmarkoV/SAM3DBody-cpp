#include "v4l2_capture.h"

// FSB_HAVE_V4L2 is defined by CMakeLists.txt only when both (a) the target
// platform is Linux and (b) <linux/videodev2.h> was found (CheckIncludeFileCXX).
// When it's not defined -- any non-Linux build, or a Linux box missing kernel
// headers -- everything below the #else is compiled instead: a thin,
// unconditional pass-through to cv::VideoCapture with the exact same public
// API, i.e. the pre-RobustCapture behavior.
#ifdef FSB_HAVE_V4L2

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

struct RobustCapture::V4L2Impl {
    int      fd = -1;
    uint32_t width  = 0;
    uint32_t height = 0;
    uint32_t fourcc = 0;      // 0 until the first G_FMT/S_FMT
    int      buffer_count = 4;
    bool     streaming = false;

    struct Buf { void* start = nullptr; size_t length = 0; };
    std::vector<Buf> buffers;

    bool     held = false;
    unsigned held_index = 0;
    size_t   held_bytesused = 0;

    ~V4L2Impl() { closeAll(); }

    bool ioctlLoop(unsigned long req, void* arg) {
        int r;
        do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
        return r != -1;
    }

    bool ensureStreaming() {
        if (streaming) return true;

        // Pick up whatever format is currently active (either the driver's
        // power-on default, or whatever an earlier set() negotiated) rather
        // than silently overriding it here.
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!ioctlLoop(VIDIOC_G_FMT, &fmt)) return false;
        width  = fmt.fmt.pix.width;
        height = fmt.fmt.pix.height;
        fourcc = fmt.fmt.pix.pixelformat;

        v4l2_requestbuffers req{};
        req.count  = (uint32_t)std::max(2, buffer_count);
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (!ioctlLoop(VIDIOC_REQBUFS, &req) || req.count < 2) return false;

        buffers.assign(req.count, Buf{});
        for (uint32_t i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (!ioctlLoop(VIDIOC_QUERYBUF, &buf)) return false;

            buffers[i].length = buf.length;
            buffers[i].start  = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, buf.m.offset);
            if (buffers[i].start == MAP_FAILED) { buffers[i].start = nullptr; return false; }
            if (!ioctlLoop(VIDIOC_QBUF, &buf)) return false;
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!ioctlLoop(VIDIOC_STREAMON, &type)) return false;
        streaming = true;
        return true;
    }

    void closeAll() {
        if (streaming) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctlLoop(VIDIOC_STREAMOFF, &type);
            streaming = false;
        }
        for (auto& b : buffers)
            if (b.start) munmap(b.start, b.length);
        buffers.clear();
        if (fd >= 0) { ::close(fd); fd = -1; }
        held = false;
    }
};

RobustCapture::RobustCapture() = default;
RobustCapture::~RobustCapture() = default;

bool RobustCapture::openV4L2(const std::string& devicePath) {
    int fd = ::open(devicePath.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) return false;

    v4l2_capability cap{};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1 ||
        !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(cap.capabilities & V4L2_CAP_STREAMING)) {
        ::close(fd);
        return false;
    }

    v4l2_ = std::make_unique<V4L2Impl>();
    v4l2_->fd = fd;
    return true;
}

bool RobustCapture::open(int index) {
    std::string dev = "/dev/video" + std::to_string(index);
    if (openV4L2(dev)) {
        use_v4l2_ = true;
        printf("[v4l2] raw V4L2 capture active for %s\n", dev.c_str());
        return true;
    }
    use_v4l2_ = false;
    return cv_cap_.open(index);
}

bool RobustCapture::open(const std::string& path) {
    bool looksLikeDevice = path.compare(0, 10, "/dev/video") == 0;
    if (looksLikeDevice && openV4L2(path)) {
        use_v4l2_ = true;
        printf("[v4l2] raw V4L2 capture active for %s\n", path.c_str());
        return true;
    }
    use_v4l2_ = false;
    return cv_cap_.open(path);
}

bool RobustCapture::isOpened() const {
    return use_v4l2_ ? (v4l2_ && v4l2_->fd >= 0) : cv_cap_.isOpened();
}

bool RobustCapture::set(int prop, double value) {
    if (!use_v4l2_) return cv_cap_.set(prop, value);
    if (!v4l2_ || v4l2_->fd < 0) return false;

    switch (prop) {
    case cv::CAP_PROP_FOURCC: {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_FMT, &fmt)) return false;
        fmt.fmt.pix.pixelformat = (uint32_t)(int64_t)value;
        if (!v4l2_->ioctlLoop(VIDIOC_S_FMT, &fmt)) return false;
        v4l2_->fourcc = fmt.fmt.pix.pixelformat;
        v4l2_->width  = fmt.fmt.pix.width;
        v4l2_->height = fmt.fmt.pix.height;
        return true;
    }
    case cv::CAP_PROP_FRAME_WIDTH: {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_FMT, &fmt)) return false;
        fmt.fmt.pix.width = (uint32_t)value;
        if (!v4l2_->ioctlLoop(VIDIOC_S_FMT, &fmt)) return false;
        v4l2_->fourcc = fmt.fmt.pix.pixelformat;
        v4l2_->width  = fmt.fmt.pix.width;
        v4l2_->height = fmt.fmt.pix.height;
        return true;
    }
    case cv::CAP_PROP_FRAME_HEIGHT: {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_FMT, &fmt)) return false;
        fmt.fmt.pix.height = (uint32_t)value;
        if (!v4l2_->ioctlLoop(VIDIOC_S_FMT, &fmt)) return false;
        v4l2_->fourcc = fmt.fmt.pix.pixelformat;
        v4l2_->width  = fmt.fmt.pix.width;
        v4l2_->height = fmt.fmt.pix.height;
        return true;
    }
    case cv::CAP_PROP_FPS: {
        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator   = 1;
        parm.parm.capture.timeperframe.denominator = (uint32_t)(value > 0 ? value : 30);
        if (!v4l2_->ioctlLoop(VIDIOC_S_PARM, &parm)) return false;
        return true;
    }
    case cv::CAP_PROP_BUFFERSIZE:
        v4l2_->buffer_count = std::max(2, (int)value);
        return true;
    default:
        // e.g. CAP_PROP_POS_FRAMES -- meaningless for a live device, ignored.
        return false;
    }
}

double RobustCapture::get(int prop) const {
    if (!use_v4l2_) return cv_cap_.get(prop);
    if (!v4l2_ || v4l2_->fd < 0) return 0.0;

    switch (prop) {
    case cv::CAP_PROP_FOURCC: {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_FMT, &fmt)) return 0.0;
        return (double)(int32_t)fmt.fmt.pix.pixelformat;
    }
    case cv::CAP_PROP_FRAME_WIDTH: {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_FMT, &fmt)) return 0.0;
        return (double)fmt.fmt.pix.width;
    }
    case cv::CAP_PROP_FRAME_HEIGHT: {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_FMT, &fmt)) return 0.0;
        return (double)fmt.fmt.pix.height;
    }
    case cv::CAP_PROP_FPS: {
        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (!v4l2_->ioctlLoop(VIDIOC_G_PARM, &parm)) return 0.0;
        auto& tpf = parm.parm.capture.timeperframe;
        if (tpf.numerator == 0) return 0.0;
        return (double)tpf.denominator / (double)tpf.numerator;
    }
    default:
        return 0.0;
    }
}

bool RobustCapture::grab() {
    if (!use_v4l2_) return cv_cap_.grab();
    if (!v4l2_) return false;
    if (!v4l2_->ensureStreaming()) return false;

    // The "discard stale frames" loop in the render loop calls grab()
    // repeatedly without an intervening retrieve() -- each call here
    // retires whichever buffer the previous grab() left held, so only the
    // newest one survives.
    if (v4l2_->held) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = v4l2_->held_index;
        v4l2_->ioctlLoop(VIDIOC_QBUF, &buf);
        v4l2_->held = false;
    }

    const int kMaxRetries = 8;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        pollfd pfd{v4l2_->fd, POLLIN, 0};
        int pr = poll(&pfd, 1, 2000);
        if (pr <= 0) return false;   // timeout/error -- treat as stream ended

        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (!v4l2_->ioctlLoop(VIDIOC_DQBUF, &buf)) {
            if (errno == EAGAIN) continue;   // spurious wakeup, try again
            return false;
        }

        if (buf.flags & V4L2_BUF_FLAG_ERROR) {
            // The kernel itself flagged this buffer's contents as
            // incomplete/corrupt (e.g. lost USB isochronous packets) --
            // this is exactly the torn/half-updated frame bug. Requeue it
            // unread and try the next one instead of handing it back.
            fprintf(stderr, "[v4l2] dropped a corrupt frame (driver set "
                             "V4L2_BUF_FLAG_ERROR), re-grabbing\n");
            v4l2_->ioctlLoop(VIDIOC_QBUF, &buf);
            continue;
        }

        v4l2_->held           = true;
        v4l2_->held_index     = buf.index;
        v4l2_->held_bytesused = buf.bytesused;
        return true;
    }
    return false;
}

bool RobustCapture::retrieve(cv::Mat& frame) {
    if (!use_v4l2_) return cv_cap_.retrieve(frame);
    if (!v4l2_ || !v4l2_->held) return false;

    auto& b = v4l2_->buffers[v4l2_->held_index];
    bool ok = true;
    switch (v4l2_->fourcc) {
    case V4L2_PIX_FMT_YUYV: {
        cv::Mat yuyv((int)v4l2_->height, (int)v4l2_->width, CV_8UC2, b.start);
        cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUYV);
        break;
    }
    case V4L2_PIX_FMT_MJPEG:
    case V4L2_PIX_FMT_JPEG: {
        cv::Mat raw(1, (int)v4l2_->held_bytesused, CV_8UC1, b.start);
        frame = cv::imdecode(raw, cv::IMREAD_COLOR);
        ok = !frame.empty();
        break;
    }
    case V4L2_PIX_FMT_BGR24: {
        cv::Mat((int)v4l2_->height, (int)v4l2_->width, CV_8UC3, b.start).copyTo(frame);
        break;
    }
    default: {
        // Unhandled raw fourcc: best-effort as YUYV, the common UVC default.
        cv::Mat yuyv((int)v4l2_->height, (int)v4l2_->width, CV_8UC2, b.start);
        cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUYV);
        break;
    }
    }

    v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = v4l2_->held_index;
    v4l2_->ioctlLoop(VIDIOC_QBUF, &buf);
    v4l2_->held = false;

    return ok;
}

bool RobustCapture::read(cv::Mat& frame) {
    if (!use_v4l2_) return cv_cap_.read(frame);

    // grab() already re-grabs frames the kernel itself flagged as corrupt
    // (V4L2_BUF_FLAG_ERROR). That flag isn't the only way a frame can come
    // out bad, though: with --mjpg, a truncated/corrupt JPEG can complete
    // its USB transfer "successfully" and just fail to decode. Retry on
    // that too, instead of surfacing an empty frame -- the render loop
    // treats an empty frame as end-of-stream and would otherwise abort the
    // whole session over a single bad JPEG frame.
    const int kMaxRetries = 8;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        if (!grab()) return false;
        if (retrieve(frame)) return true;
        fprintf(stderr, "[v4l2] dropped a frame that failed to decode, re-grabbing\n");
    }
    return false;
}

RobustCapture& RobustCapture::operator>>(cv::Mat& frame) {
    if (!read(frame)) frame.release();
    return *this;
}

void RobustCapture::release() {
    if (use_v4l2_) {
        v4l2_.reset();
        use_v4l2_ = false;
    } else {
        cv_cap_.release();
    }
}

#else  // !FSB_HAVE_V4L2 -- plain cv::VideoCapture pass-through, no raw V4L2 anywhere.

// V4L2Impl must exist (even if unused) so unique_ptr<V4L2Impl>'s destructor
// has a complete type to destroy; it's never actually allocated below.
struct RobustCapture::V4L2Impl {};

RobustCapture::RobustCapture() = default;
RobustCapture::~RobustCapture() = default;

bool RobustCapture::openV4L2(const std::string&) { return false; }

bool RobustCapture::open(int index) {
    use_v4l2_ = false;
    return cv_cap_.open(index);
}

bool RobustCapture::open(const std::string& path) {
    use_v4l2_ = false;
    return cv_cap_.open(path);
}

bool RobustCapture::isOpened() const { return cv_cap_.isOpened(); }
bool RobustCapture::set(int prop, double value) { return cv_cap_.set(prop, value); }
double RobustCapture::get(int prop) const { return cv_cap_.get(prop); }
bool RobustCapture::grab() { return cv_cap_.grab(); }
bool RobustCapture::retrieve(cv::Mat& frame) { return cv_cap_.retrieve(frame); }
bool RobustCapture::read(cv::Mat& frame) { return cv_cap_.read(frame); }

RobustCapture& RobustCapture::operator>>(cv::Mat& frame) {
    cv_cap_.read(frame);
    return *this;
}

void RobustCapture::release() { cv_cap_.release(); }

#endif // FSB_HAVE_V4L2
