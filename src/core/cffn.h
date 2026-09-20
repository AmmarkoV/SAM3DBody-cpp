#pragma once
// ============================================================================
// cffn.h  –  dense layer kernel for the CPU MHR / camera FFN heads
//
// Weights are stored TRANSPOSED, [in, out] row-major (wt[k * out + n] =
// weight from input k to output n), rather than the [out, in] layout they
// have in the GGUF.  That puts the inner loop over independent outputs, so it
// vectorises without -ffast-math — the textbook dot-product form is a serial
// float reduction GCC will not reorder.  And with k outermost the weights
// (~6 MB for the MHR head) are streamed once per batch instead of once per
// person.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <vector>

namespace fsb {

// [rows, cols] row-major -> [cols, rows] row-major
inline std::vector<float> transpose_rows(const std::vector<float>& w, int rows, int cols)
{
    std::vector<float> t(w.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            t[(size_t)c * rows + r] = w[(size_t)r * cols + c];
    return t;
}

// y = x @ wt + b, optionally ReLU'd.   x:[B,K]  wt:[K,N]  b:[N]  → y:[B,N]
inline void linear_relu(const float* __restrict x, const float* __restrict wt,
                        const float* __restrict b, float* __restrict y,
                        int B, int K, int N, bool relu)
{
    for (int bi = 0; bi < B; ++bi)
        std::copy(b, b + N, y + (size_t)bi * N);

    for (int k = 0; k < K; ++k)
    {
        const float* wr = wt + (size_t)k * N;
        for (int bi = 0; bi < B; ++bi)
        {
            const float xk = x[(size_t)bi * K + k];
            float*      yr = y + (size_t)bi * N;
            for (int n = 0; n < N; ++n) yr[n] += xk * wr[n];
        }
    }

    if (relu)
        for (size_t i = 0; i < (size_t)B * N; ++i) y[i] = std::max(0.f, y[i]);
}

} // namespace fsb
