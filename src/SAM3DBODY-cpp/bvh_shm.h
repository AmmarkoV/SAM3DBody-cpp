/** @file bvh_shm.h
 *  @brief Publish per-frame BVH channel floats over POSIX shared memory
 *         (AmmarkoV/SharedMemoryVideoBuffers) for the live webcam→robot pipe.
 *
 *  This is the zero-copy counterpart of the "@F <floats>" stdout line: instead
 *  of serialising the channels to ASCII and having the Python consumer write a
 *  temp .bvh and re-parse it every frame, we drop the raw float array into a
 *  shared-memory "generic" buffer.  The buffer's timestamp field is (ab)used as
 *  a MONOTONIC FRAME COUNTER — the library's real timestamps are only
 *  seconds-resolution, too coarse to tell consecutive frames apart, so the
 *  consumer polls this counter to detect a new frame.
 *
 *  Compiled only when FSB_SHM is defined (Linux; see CMakeLists.txt).  The
 *  header itself is always includable — the class simply never opens on a build
 *  without the library, so callers can guard purely on is_open().
 */
#ifndef FSB_BVH_SHM_H_INCLUDED
#define FSB_BVH_SHM_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class BvhShmPublisher
{
public:
    BvhShmPublisher() = default;
    ~BvhShmPublisher() { close(); }

    BvhShmPublisher(const BvhShmPublisher&)            = delete;
    BvhShmPublisher& operator=(const BvhShmPublisher&) = delete;

    /** Create/attach the shared-memory descriptor and remember the feed name.
     *  @param descriptor POSIX shm object name (e.g. "sam3dbody_bvh.shm").
     *  @param stream     feed name within the descriptor (e.g. "bvh").
     *  @return true if the descriptor was attached (or the build lacks FSB_SHM
     *          → false, so the caller falls back to the stdout stream).       */
    bool open(const std::string& descriptor, const std::string& stream);

    /** Publish one frame of channel floats.  The generic buffer is created
     *  lazily on the first call, sized to floats.size().  Subsequent frames
     *  must carry the same count.  @return false on any hard error.           */
    bool publish(const std::vector<float>& floats);

    bool is_open() const { return ctx_ != nullptr; }
    void close();

private:
    void*       ctx_     = nullptr;  // struct SharedMemoryContext*
    void*       frame_   = nullptr;  // struct VideoFrame*
    std::string descriptor_;
    std::string stream_;
    std::size_t n_       = 0;        // channel count, locked on first publish
    std::uint64_t counter_ = 0;      // monotonic frame counter (→ timestamp field)
};

#endif // FSB_BVH_SHM_H_INCLUDED
