// core_test.cpp — deterministic checks for the pure helpers in src/core.
//   * linear_relu (cffn.h) against the textbook y = relu(x @ W.T + b) it replaced
//   * bbox_iou (bbox_iou.h)
//   * build_model_params (preprocess.hpp) layout
// Exits 0 on success, 1 on failure.

#include "cffn.h"
#include "bbox_iou.h"
#include "preprocess.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                              printf(__VA_ARGS__); printf("\n"); } } while (0)

// The pre-cffn.h kernel: W in its GGUF [out, in] layout, one dot product per output.
static void linear_relu_reference(const float* x, const float* w, const float* b,
                                  float* y, int B, int K, int N, bool relu)
{
    for (int bi = 0; bi < B; ++bi)
        for (int n = 0; n < N; ++n)
        {
            float s = b[n];
            for (int k = 0; k < K; ++k) s += x[bi * K + k] * w[n * K + k];
            y[bi * N + n] = relu ? std::max(0.f, s) : s;
        }
}

static void test_linear_relu()
{
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int B : {1, 2, 5})
        for (bool relu : {false, true})
        {
            // Odd sizes on purpose, so no vector-width tail is skipped.
            const int K = 1027, N = 519;
            std::vector<float> x((size_t)B * K), w((size_t)N * K), b(N);
            for (auto& v : x) v = nd(rng);
            for (auto& v : w) v = nd(rng) * 0.03f;
            for (auto& v : b) v = nd(rng);

            std::vector<float> want((size_t)B * N), got((size_t)B * N);
            linear_relu_reference(x.data(), w.data(), b.data(), want.data(), B, K, N, relu);
            const std::vector<float> wt = fsb::transpose_rows(w, N, K);
            fsb::linear_relu(x.data(), wt.data(), b.data(), got.data(), B, K, N, relu);

            // Only the summation order differs, so allow float reassociation error.
            float worst = 0.f;
            for (size_t i = 0; i < want.size(); ++i)
                worst = std::max(worst, std::fabs(want[i] - got[i]) / (1.f + std::fabs(want[i])));
            CHECK(worst < 1e-5f, "linear_relu B=%d relu=%d: rel err %g", B, relu, worst);
        }
}

static void test_bbox_iou()
{
    const float a[4] = {0, 0, 10, 10};
    const float same[4] = {0, 0, 10, 10};
    const float half[4] = {5, 0, 15, 10};       // inter 50, union 150
    const float apart[4] = {20, 20, 30, 30};
    const float touch[4] = {10, 0, 20, 10};     // shares an edge only
    const float flipped[4] = {10, 10, 0, 0};    // x2 < x1: empty box
    CHECK(std::fabs(fsb::bbox_iou(a, same) - 1.f) < 1e-6f, "identical boxes");
    CHECK(std::fabs(fsb::bbox_iou(a, half) - 1.f / 3.f) < 1e-6f, "half overlap");
    CHECK(fsb::bbox_iou(a, apart) == 0.f, "disjoint boxes");
    CHECK(fsb::bbox_iou(a, touch) == 0.f, "edge-touching boxes");
    CHECK(fsb::bbox_iou(a, flipped) == 0.f, "degenerate box");
    CHECK(fsb::bbox_iou(half, a) == fsb::bbox_iou(a, half), "symmetric");
    const std::array<float, 4> aa{0, 0, 10, 10}, hh{5, 0, 15, 10};
    CHECK(fsb::bbox_iou(aa, hh) == fsb::bbox_iou(a, half), "std::array overload");
}

static void test_build_model_params()
{
    const float rot[3] = {0.1f, 0.2f, 0.3f};          // rot6d_to_euler order: rx, ry, rz
    float body[133];
    for (int i = 0; i < 133; ++i) body[i] = 1.f + i;  // all non-zero

    const fsb::ModelParams204 mp = fsb::build_model_params(rot, body);
    CHECK(mp.data[0] == 0.f && mp.data[1] == 0.f && mp.data[2] == 0.f, "global_trans zeroed");
    CHECK(mp.data[3] == rot[2] && mp.data[4] == rot[1] && mp.data[5] == rot[0],
          "global rot stored as (rz, ry, rx)");
    for (int i = 6; i < 136; ++i)
    {
        const bool hand = (i >= 68 && i <= 121);
        const float want = hand ? 0.f : body[i - 6];
        CHECK(mp.data[i] == want, "model_params[%d] = %g, want %g", i, mp.data[i], want);
    }
    for (int i = 136; i < 204; ++i)
        CHECK(mp.data[i] == 0.f, "scale slot %d not zero", i);
}

int main()
{
    test_linear_relu();
    test_bbox_iou();
    test_build_model_params();
    if (g_fail) { printf("core_test: %d failure(s)\n", g_fail); return 1; }
    printf("core_test: all passed\n");
    return 0;
}
