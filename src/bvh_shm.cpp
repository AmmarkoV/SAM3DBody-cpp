/** @file bvh_shm.cpp  — see bvh_shm.h.  Only meaningful under FSB_SHM. */
#include "bvh_shm.h"

#ifdef FSB_SHM
extern "C" {
#include "sharedMemoryVideoBuffers.h"   // vendored: SharedMemoryVideoBuffers/src/c
}
#include <cstdio>

// The library uses two return conventions: create*/EXIT_SUCCESS is 0-on-success,
// while start/stopWriting return 1-on-success.  Casts below keep the opaque
// void* handles out of the header so callers never need the C struct layout.

bool BvhShmPublisher::open(const std::string& descriptor, const std::string& stream)
{
    descriptor_ = descriptor;
    stream_     = stream;

    // Create the descriptor (idempotent shm_open(O_CREAT); zero-inits the ctx).
    // We are the first process up, so this is safe before the consumer attaches.
    if (createSharedMemoryContextDescriptor(descriptor_.c_str()) == -1)
    {
        std::fprintf(stderr, "[bvh_shm] could not create shm descriptor '%s'\n", descriptor_.c_str());
        return false;
    }
    ctx_ = connectToSharedMemoryContextDescriptor(descriptor_.c_str());
    if (!ctx_)
    {
        std::fprintf(stderr, "[bvh_shm] could not connect to shm descriptor '%s'\n", descriptor_.c_str());
        return false;
    }
    return true;
}

bool BvhShmPublisher::publish(const std::vector<float>& floats)
{
    if (!ctx_ || floats.empty()) return false;

    if (n_ == 0)   // first frame: size + create the generic buffer, then map it
    {
        n_ = floats.size();
        const unsigned int bytes = static_cast<unsigned int>(n_ * sizeof(float));
        if (createGenericMetaData(static_cast<SharedMemoryContext*>(ctx_),
                                  stream_.c_str(), bytes) != EXIT_SUCCESS)
        {
            std::fprintf(stderr, "[bvh_shm] createGenericMetaData('%s', %u B) failed\n",
                         stream_.c_str(), bytes);
            ctx_ = nullptr;
            return false;
        }
        frame_ = getVideoBufferPointer(static_cast<SharedMemoryContext*>(ctx_), stream_.c_str());
        if (!frame_ || map_frame_shared_memory(static_cast<VideoFrame*>(frame_), 1) == nullptr)
        {
            std::fprintf(stderr, "[bvh_shm] could not map shm feed '%s'\n", stream_.c_str());
            frame_ = nullptr;
            ctx_   = nullptr;
            return false;
        }
        std::fprintf(stderr, "[bvh_shm] publishing %zu-float frames to '%s:%s'\n",
                     n_, descriptor_.c_str(), stream_.c_str());
    }

    if (floats.size() != n_) return false;   // channel count must stay constant

    VideoFrame* vf = static_cast<VideoFrame*>(frame_);
    if (!startWritingToVideoBufferPointer(vf)) return false;
    copy_to_shared_memory(vf, floats.data(), n_ * sizeof(float),
                          static_cast<unsigned long>(++counter_));
    stopWritingToVideoBufferPointer(vf);
    return true;
}

void BvhShmPublisher::close()
{
    if (ctx_ && !stream_.empty())
        destroyVideoFrame(static_cast<SharedMemoryContext*>(ctx_), stream_.c_str());
    ctx_   = nullptr;
    frame_ = nullptr;
}

#else  // !FSB_SHM — no shared-memory library compiled in; publisher is a no-op.

bool BvhShmPublisher::open(const std::string&, const std::string&) { return false; }
bool BvhShmPublisher::publish(const std::vector<float>&)           { return false; }
void BvhShmPublisher::close()                                      {}

#endif // FSB_SHM
