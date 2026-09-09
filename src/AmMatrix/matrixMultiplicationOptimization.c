#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#if INTEL_OPTIMIZATIONS
 #include <immintrin.h>
 //#include <intrin.h>
 #include <x86intrin.h>
#endif


void mmul_sse(const float * a, const float * b, float * r)
{
#if INTEL_OPTIMIZATIONS
  __m128 a_line, b_line, r_line;
  for (int i=0; i<16; i+=4) {
    // unroll the first step of the loop to avoid having to initialize r_line to zero
    a_line = _mm_load_ps(a);         // a_line = vec4(column(a, 0))
    b_line = _mm_set1_ps(b[i]);      // b_line = vec4(b[i][0])
    r_line = _mm_mul_ps(a_line, b_line); // r_line = a_line * b_line
    for (int j=1; j<4; j++) {
      a_line = _mm_load_ps(&a[j*4]); // a_line = vec4(column(a, j))
      b_line = _mm_set1_ps(b[i+j]);  // b_line = vec4(b[i][j])
                                     // r_line += a_line * b_line
      r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
    }
    _mm_store_ps(&r[i], r_line);     // r[i] = r_line
  }
#else
  fprintf(stderr,"mmul_sse not available in this build/architecture\n");
#endif
}


/*
union Mat44 {
    float m[4][4];
    __m128 row[4];
};

// reference implementation
void matmult_ref(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    Mat44 t; // write to temp
    for (int i=0; i < 4; i++)
        for (int j=0; j < 4; j++)
            t.m[i][j] = A.m[i][0]*B.m[0][j] + A.m[i][1]*B.m[1][j] + A.m[i][2]*B.m[2][j] + A.m[i][3]*B.m[3][j];

    out = t;
}

// linear combination:
// a[0] * B.row[0] + a[1] * B.row[1] + a[2] * B.row[2] + a[3] * B.row[3]
static inline __m128 lincomb_SSE(const __m128 &a, const Mat44 &B)
{
    __m128 result;
    result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), B.row[0]);
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x55), B.row[1]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xaa), B.row[2]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xff), B.row[3]));
    return result;
}

// this is the right approach for SSE ... SSE4.2
void matmult_SSE(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    // out_ij = sum_k a_ik b_kj
    // => out_0j = a_00 * b_0j + a_01 * b_1j + a_02 * b_2j + a_03 * b_3j
    __m128 out0x = lincomb_SSE(A.row[0], B);
    __m128 out1x = lincomb_SSE(A.row[1], B);
    __m128 out2x = lincomb_SSE(A.row[2], B);
    __m128 out3x = lincomb_SSE(A.row[3], B);

    out.row[0] = out0x;
    out.row[1] = out1x;
    out.row[2] = out2x;
    out.row[3] = out3x;
}

// another linear combination, using AVX instructions on XMM regs
static inline __m128 lincomb_AVX_4mem(const float *a, const Mat44 &B)
{
    __m128 result;
    result = _mm_mul_ps(_mm_broadcast_ss(&a[0]), B.row[0]);
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[1]), B.row[1]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[2]), B.row[2]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[3]), B.row[3]));
    return result;
}

// using AVX instructions, 4-wide
// this can be better if A is in memory.
void matmult_AVX_4mem(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    _mm256_zeroupper();
    __m128 out0x = lincomb_AVX_4mem(A.m[0], B);
    __m128 out1x = lincomb_AVX_4mem(A.m[1], B);
    __m128 out2x = lincomb_AVX_4mem(A.m[2], B);
    __m128 out3x = lincomb_AVX_4mem(A.m[3], B);

    out.row[0] = out0x;
    out.row[1] = out1x;
    out.row[2] = out2x;
    out.row[3] = out3x;
}

// dual linear combination using AVX instructions on YMM regs
static inline __m256 twolincomb_AVX_8(__m256 A01, const Mat44 &B)
{
    __m256 result;
    result = _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0x00), _mm256_broadcast_ps(&B.row[0]));
    result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0x55), _mm256_broadcast_ps(&B.row[1])));
    result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0xaa), _mm256_broadcast_ps(&B.row[2])));
    result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0xff), _mm256_broadcast_ps(&B.row[3])));
    return result;
}

// this should be noticeably faster with actual 256-bit wide vector units (Intel);
// not sure about double-pumped 128-bit (AMD), would need to check.
void matmult_AVX_8(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    _mm256_zeroupper();
    __m256 A01 = _mm256_loadu_ps(&A.m[0][0]);
    __m256 A23 = _mm256_loadu_ps(&A.m[2][0]);
    
    __m256 out01x = twolincomb_AVX_8(A01, B);
    __m256 out23x = twolincomb_AVX_8(A23, B);

    _mm256_storeu_ps(&out.m[0][0], out01x);
    _mm256_storeu_ps(&out.m[2][0], out23x);
}

// ---- testing stuff

static float randf()
{
    // assumes VC++ rand()
    return (rand() - 16384.0f) / 1024.0f;
}

static void randmat(Mat44 &M)
{
    for (int i=0; i < 4; i++)
        for (int j=0; j < 4; j++)
            M.m[i][j] = randf();
}

int the_mask = 0; // global so the compiler can't be sure what its value is for opt.

static void run_ref(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_ref(out[j], A[j], B[j]);
    }
}

static void run_SSE(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_SSE(out[j], A[j], B[j]);
    }
}

static void run_AVX_4mem(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_AVX_4mem(out[j], A[j], B[j]);
    }
}

static void run_AVX_8(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_AVX_8(out[j], A[j], B[j]);
    }
}

int main(int argc, char **argv)
{
    static const struct {
        const char *name;
        void (*matmult)(Mat44 &out, const Mat44 &A, const Mat44 &B);
    } variants[] = {
        { "ref",      matmult_ref },
        { "SSE",      matmult_SSE },
        { "AVX_4mem", matmult_AVX_4mem },
        { "AVX_8",    matmult_AVX_8 },
    };
    static const int nvars = (int) (sizeof(variants) / sizeof(*variants));
    
    srand(1234); // deterministic random tests(TM)

    // correctness tests
    // when compiled with /arch:SSE (or SSE2/AVX), all functions are
    // supposed to return the exact same results!
    for (int i=0; i < 1000000; i++)
    {
        Mat44 A, B, out, ref_out;
        randmat(A);
        randmat(B);
        matmult_ref(ref_out, A, B);

        for (int j=0; j < nvars; j++)
        {
            variants[j].matmult(out, A, B);
            if (memcmp(&out, &ref_out, sizeof(out)) != 0)
            {
                fprintf(stderr, "%s fails test\n", variants[j].name);
                exit(1);
            }
        }
    }

    printf("all ok.\n");

    // perf tests
    // as usual with such microbenchmarks, this isn't measuring anything
    // terribly useful, but here goes.
    static const struct {
        const char *name;
        void (*run)(Mat44 *out, const Mat44 *A, const Mat44 *B, int count);
    } perf_variants[] = {
        { "ref",      run_ref },
        { "SSE",      run_SSE },
        { "AVX_4mem", run_AVX_4mem },
        { "AVX_8",    run_AVX_8 },
    };
    static const int nperfvars = (int) (sizeof(perf_variants) / sizeof(*perf_variants));
    
    
    Mat44 Aperf, Bperf, out;
    randmat(Aperf);
    randmat(Bperf);

    for (int i=0; i < nvars; i++)
    {
        static const int nruns = 4096;
        static const int muls_per_run = 4096;
        unsigned long long best_time = ~0ull;

        for (int run=0; run < nruns; run++)
        {
            unsigned long long time = __rdtsc();
            perf_variants[i].run(&out, &Aperf, &Bperf, muls_per_run);
            time = __rdtsc() - time;
            if (time < best_time)
                best_time = time;
        }

        double cycles_per_run = (double) best_time / (double) muls_per_run;
        printf("%12s: %.2f cycles\n", perf_variants[i].name, cycles_per_run);
    }

    return 0;
}

*/
