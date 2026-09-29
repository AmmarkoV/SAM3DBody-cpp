#include "shm_reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cstring>

namespace h1b {

int64_t mono_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);   // same clock as Python's time.monotonic_ns()
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

TeleopShmReader::~TeleopShmReader() { close(); }

void TeleopShmReader::close() {
    if (map_) munmap(map_, sizeof(TeleopShm));
    map_ = nullptr;
    mapped_ = false;
    ino_ = 0;
}

bool TeleopShmReader::stale() const {
    if (!map_) return true;
    struct stat sb;
    if (stat(("/dev/shm/" + name_).c_str(), &sb) != 0) return true;
    return sb.st_ino != ino_;
}

bool TeleopShmReader::connect(const std::string& name, const std::string& expect_robot, std::string* err) {
    if (map_ && !stale() && mapped_) return true;
    close();
    name_ = name;
    const std::string path = "/dev/shm/" + name;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { *err = "waiting for " + path; return false; }
    struct stat sb;
    if (fstat(fd, &sb) != 0 || size_t(sb.st_size) < sizeof(TeleopShm)) {
        ::close(fd);
        *err = path + " too small";
        return false;
    }
    void* p = mmap(nullptr, sizeof(TeleopShm), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { *err = "mmap failed"; return false; }
    map_ = static_cast<TeleopShm*>(p);
    ino_ = sb.st_ino;

    const TeleopShmHeader& h = map_->header;
    if (__atomic_load_n(&h.magic, __ATOMIC_ACQUIRE) != TELEOP_SHM_MAGIC) { *err = "header not ready"; close(); return false; }
    if (h.version != TELEOP_SHM_VERSION) { *err = "shm version mismatch"; close(); return false; }
    if (h.n_joints > TELEOP_MAX_JOINTS) { *err = "bad joint count"; close(); return false; }
    char robot[TELEOP_NAME_LEN + 1] = {};
    std::memcpy(robot, h.robot, TELEOP_NAME_LEN);
    if (expect_robot != robot) {
        *err = "shm is for robot '" + std::string(robot) + "', expected '" + expect_robot + "'";
        close();
        return false;
    }
    // Map by NAME: refuse on any missing joint rather than trust an index order.
    for (int j = 0; j < kNumJoints; ++j) {
        index_[j] = -1;
        for (uint32_t i = 0; i < h.n_joints; ++i) {
            char n[TELEOP_NAME_LEN + 1] = {};
            std::memcpy(n, h.joint_names[i], TELEOP_NAME_LEN);
            if (std::strcmp(n, kJoints[j].name) == 0) { index_[j] = int(i); break; }
        }
        if (index_[j] < 0) {
            *err = std::string("shm has no joint '") + kJoints[j].name + "'";
            close();
            return false;
        }
    }
    mapped_ = true;
    return true;
}

bool TeleopShmReader::read(int64_t now_mono_ns, TargetIn* out) {
    if (!connected()) return false;
    const TeleopShmFrame* f = &map_->frame;
    TeleopShmFrame local;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const uint64_t s1 = __atomic_load_n(&f->seq, __ATOMIC_ACQUIRE);
        if (s1 & 1u) continue;                              // writer mid-update
        std::memcpy(&local, (const void*)f, sizeof(local));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const uint64_t s2 = __atomic_load_n(&f->seq, __ATOMIC_ACQUIRE);
        if (s1 != s2) continue;                             // torn
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&local);
        if (teleop_crc32(bytes + TELEOP_CRC_OFFSET, sizeof(local) - TELEOP_CRC_OFFSET) != local.crc32)
            continue;                                       // corrupted
        if (local.counter == 0) return false;               // nothing published yet
        out->have = true;
        out->counter = local.counter;
        out->flags = local.flags;
        const double age = double(now_mono_ns - local.t_mono_ns) * 1e-9;
        out->age_s = age >= -0.01 ? (age < 0 ? 0.0 : age) : 1e9;   // future stamp = distrust
        for (int j = 0; j < kNumJoints; ++j) out->q[j] = local.q[index_[j]];
        return true;
    }
    return false;
}

}  // namespace h1b
