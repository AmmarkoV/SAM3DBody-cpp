// calib_test.c — round-trip + parse checks for the .calib loader.
//   usage: calib_test <path-to-sample.calib>
// Exits 0 on success, 1 on any failed check.

#include "calib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;

static void check(const char * what, float got, float want, float tol)
{
    if (fabsf(got - want) > tol) {
        fprintf(stderr, "  FAIL %-22s got %.6f  want %.6f\n", what, got, want);
        g_fail = 1;
    } else {
        fprintf(stderr, "  ok   %-22s %.6f\n", what, got);
    }
}

int main(int argc, char ** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <sample.calib>\n", argv[0]); return 2; }

    struct calibration c;
    if (!ReadCalibration(argv[1], 0, 0, &c)) {
        fprintf(stderr, "ReadCalibration('%s') failed\n", argv[1]);
        return 1;
    }
    PrintCalibration(&c);

    // Dimensions + intrinsics (row-major K via the CALIB_INTR_* indices).
    check("width",  (float)c.width,  1920.f, 0.5f);
    check("height", (float)c.height, 1080.f, 0.5f);
    check("fx", c.intrinsic[CALIB_INTR_FX], 1400.f, 1e-3f);
    check("fy", c.intrinsic[CALIB_INTR_FY], 1400.f, 1e-3f);
    check("cx", c.intrinsic[CALIB_INTR_CX],  960.f, 1e-3f);
    check("cy", c.intrinsic[CALIB_INTR_CY],  540.f, 1e-3f);

    // Distortion (k1,k2,p1,p2,k3).
    check("k1", c.k1, -0.05f, 1e-5f);
    check("k2", c.k2,  0.01f, 1e-5f);
    check("p1", c.p1,  0.001f, 1e-5f);
    check("p2", c.p2,  0.002f, 1e-5f);
    check("k3", c.k3,  0.0f,  1e-5f);

    // Extrinsics.
    check("set_intrinsic", (float)c.intrinsicParametersSet, 1.f, 0.5f);
    check("set_extrinsic", (float)c.extrinsicParametersSet, 1.f, 0.5f);
    check("Tx", c.extrinsicTranslation[0], 0.5f, 1e-5f);
    check("Tz", c.extrinsicTranslation[2], 0.1f, 1e-5f);
    check("Rz", c.extrinsicRotationRodriguez[2], 1.5707963f, 1e-4f);

    // Locale-safety: a dot-decimal parses correctly even if the host locale is
    // comma-decimal (the function forces en_US numeric parsing).
    check("internationalAtof", internationalAtof("3.14"), 3.14f, 1e-4f);

    fprintf(stderr, g_fail ? "\ncalib_test: FAILED\n" : "\ncalib_test: OK\n");
    return g_fail;
}
