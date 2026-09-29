// shm_reader.h — consumer side of robot/teleop_shm.h for the bridge.
#pragma once
#include <string>

#include "../teleop_shm.h"
#include "safety.h"

namespace h1b {

class TeleopShmReader {
public:
    ~TeleopShmReader();

    // Opens (or re-opens, if the writer restarted and recreated the object) the shm.
    // Returns true when a complete header for `expect_robot` is mapped and every
    // bridge joint was found by name.  On failure `err` says why; call again later.
    bool connect(const std::string& name, const std::string& expect_robot, std::string* err);
    bool connected() const { return map_ != nullptr && mapped_; }
    // True if /dev/shm/<name> now refers to a different object than the one mapped.
    bool stale() const;

    // Latest consistent frame (seqlock + CRC verified), remapped to bridge joint order.
    // Returns false if nothing consistent could be read this call.
    bool read(int64_t now_mono_ns, TargetIn* out);

    void close();

private:
    std::string name_;
    TeleopShm* map_ = nullptr;
    unsigned long ino_ = 0;
    bool mapped_ = false;
    int index_[kNumJoints] = {};     // bridge joint j -> q[] index in the shm
};

int64_t mono_now_ns();

}  // namespace h1b
