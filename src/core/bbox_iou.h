#pragma once
// ============================================================================
// bbox_iou.h  –  intersection-over-union of two axis-aligned boxes
//
// Boxes are [x1, y1, x2, y2].  A box with x2 < x1 or y2 < y1 counts as empty
// area rather than negative area, and disjoint or empty boxes give 0.
// ============================================================================

#include <algorithm>
#include <array>

namespace fsb {

inline float bbox_iou(const float a[4], const float b[4])
{
    const float iw = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float ih = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = iw * ih;
    if (inter <= 0.f) return 0.f;
    const float aa = std::max(0.f, a[2] - a[0]) * std::max(0.f, a[3] - a[1]);
    const float bb = std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
    const float u  = aa + bb - inter;
    return u > 0.f ? inter / u : 0.f;
}

inline float bbox_iou(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    return bbox_iou(a.data(), b.data());
}

} // namespace fsb
