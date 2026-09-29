// teleop_shm.h — retargeted-pose hand-off from tools/gmr_stream.py to a robot bridge.
//
// One POSIX shm object (/dev/shm/<name>) holding a static header plus the latest
// frame.  Deliberately tiny and self-contained (no SharedMemoryVideoBuffers): the
// bridge is the safety authority, so its input format must be auditable at a glance.
//
//   * the header lists the joint NAMES in q[] order, so the bridge maps by name and
//     refuses to arm on any mismatch instead of trusting an index convention;
//   * frames are published under a seqlock (seq odd while writing) AND carry a CRC-32
//     of their payload, so a torn or corrupted read is always detected;
//   * t_mono_ns is CLOCK_MONOTONIC on the writer; both processes run on the same
//     machine, so the reader can compute the frame's true age.
//
// The Python mirror is tools/teleop_shm.py; both sides pin the offsets below.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define TELEOP_SHM_MAGIC   0x314C5054u   /* "TPL1" */
#define TELEOP_SHM_VERSION 1u
#define TELEOP_MAX_JOINTS  64
#define TELEOP_NAME_LEN    32
#define TELEOP_HAND_CH     12            /* per hand; semantics are hand-specific */

enum {
    TELEOP_F_TRACKING    = 1u << 0,  /* q[] is a fresh retarget of a tracked person */
    TELEOP_F_HANDS_VALID = 1u << 1,  /* hand[][] is meaningful                       */
    TELEOP_F_SHUTDOWN    = 1u << 2,  /* the writer is exiting: ramp out now          */
};

typedef struct {
    uint32_t magic;                  /* written LAST; 0 until the header is complete */
    uint32_t version;
    uint32_t n_joints;
    uint32_t n_hand_ch;
    char     robot[TELEOP_NAME_LEN];
    char     joint_names[TELEOP_MAX_JOINTS][TELEOP_NAME_LEN];
    int64_t  writer_pid;
} TeleopShmHeader;

typedef struct {
    uint64_t seq;                    /* seqlock: odd while the writer is mid-update   */
    uint32_t crc32;                  /* zlib CRC-32 of bytes [16, sizeof) of the frame */
    uint32_t _pad0;
    uint64_t counter;                /* strictly increasing per published frame        */
    int64_t  t_mono_ns;              /* CLOCK_MONOTONIC at publish                     */
    uint32_t flags;                  /* TELEOP_F_*                                     */
    uint32_t _pad1;
    float    root[7];                /* root pos xyz (m) + quat wxyz (GMR world)       */
    float    _pad2;
    float    q[TELEOP_MAX_JOINTS];   /* joint angles (rad), order = header.joint_names */
    float    hand[2][TELEOP_HAND_CH];/* [left, right]                                  */
} TeleopShmFrame;

typedef struct {
    TeleopShmHeader header;
    TeleopShmFrame  frame;
} TeleopShm;

#define TELEOP_CRC_OFFSET 16u

#ifdef __cplusplus
static_assert(sizeof(TeleopShmHeader) == 2104, "TeleopShmHeader layout");
static_assert(offsetof(TeleopShmFrame, counter) == 16, "frame layout");
static_assert(offsetof(TeleopShmFrame, flags) == 32, "frame layout");
static_assert(offsetof(TeleopShmFrame, root) == 40, "frame layout");
static_assert(offsetof(TeleopShmFrame, q) == 72, "frame layout");
static_assert(offsetof(TeleopShmFrame, hand) == 328, "frame layout");
static_assert(sizeof(TeleopShmFrame) == 424, "frame layout");
static_assert(offsetof(TeleopShm, frame) == 2104, "shm layout");
static_assert(sizeof(TeleopShm) == 2528, "shm layout");
#endif

/* zlib-compatible CRC-32 (reflected, poly 0xEDB88320) — matches Python zlib.crc32. */
static inline uint32_t teleop_crc32(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}
