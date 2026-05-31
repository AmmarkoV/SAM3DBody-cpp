#include <check.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>

/*
 * Security invariant: When computing allocation sizes for matrix operations,
 * the product of dimensions must never overflow, and the allocated buffer
 * must always be large enough to hold all elements written to it.
 *
 * Specifically: pointsNum * elements * sizeof(double) must not overflow,
 * and any subsequent writes must stay within the allocated bounds.
 */

/* Safe multiplication that detects overflow */
static int safe_multiply_size(size_t a, size_t b, size_t *result) {
    if (a == 0 || b == 0) {
        *result = 0;
        return 1; /* success */
    }
    if (a > SIZE_MAX / b) {
        return 0; /* overflow detected */
    }
    *result = a * b;
    return 1; /* success */
}

/* Simulate the allocation logic from solveLinearSystemGJ.c with overflow check */
static double *safe_alloc_matrix(size_t pointsNum, size_t elements) {
    size_t total_elements;
    size_t total_bytes;

    /* Check pointsNum * elements overflow */
    if (!safe_multiply_size(pointsNum, elements, &total_elements)) {
        return NULL; /* overflow detected, refuse allocation */
    }

    /* Check total_elements * sizeof(double) overflow */
    if (!safe_multiply_size(total_elements, sizeof(double), &total_bytes)) {
        return NULL; /* overflow detected, refuse allocation */
    }

    /* Sanity check: refuse absurdly large allocations */
    if (total_bytes > (size_t)1024 * 1024 * 1024) {
        return NULL; /* too large, refuse */
    }

    if (total_bytes == 0) {
        return NULL;
    }

    return (double *)malloc(total_bytes);
}

/* Verify that if allocation succeeds, the buffer is large enough for all writes */
static int verify_buffer_bounds(double *buf, size_t pointsNum, size_t elements) {
    if (buf == NULL) return 1; /* NULL is safe - no writes possible */

    size_t total_elements;
    size_t total_bytes;

    if (!safe_multiply_size(pointsNum, elements, &total_elements)) return 0;
    if (!safe_multiply_size(total_elements, sizeof(double), &total_bytes)) return 0;

    /* Verify we can safely access all indices that would be written */
    /* Write a canary pattern to verify bounds */
    for (size_t i = 0; i < total_elements; i++) {
        buf[i] = (double)i; /* This must not overflow the buffer */
    }

    /* Verify the canary pattern */
    for (size_t i = 0; i < total_elements; i++) {
        if (buf[i] != (double)i) return 0;
    }

    return 1;
}

START_TEST(test_matrix_allocation_no_overflow)
{
    /* Invariant: Matrix allocation must never overflow size calculations,
     * and allocated buffers must always be large enough for all writes */

    struct {
        size_t pointsNum;
        size_t elements;
        const char *description;
    } adversarial_cases[] = {
        /* Normal valid cases */
        { 4, 4, "normal 4x4 matrix" },
        { 1, 1, "minimal 1x1 matrix" },
        { 10, 5, "10x5 matrix" },
        { 100, 100, "100x100 matrix" },

        /* Boundary values that could cause overflow on 32-bit */
        { 65536, 65536, "64K x 64K - likely overflow" },
        { 0xFFFF, 0xFFFF, "max 16-bit x max 16-bit" },
        { 0x10000, 0x10000, "64K x 64K boundary" },

        /* Values that overflow 32-bit size_t multiplication */
        { 0xFFFFFFFF, 2, "max 32-bit x 2 - overflow on 32-bit" },
        { 2, 0xFFFFFFFF, "2 x max 32-bit - overflow on 32-bit" },
        { 0x80000000, 2, "half max 32-bit x 2 - overflow on 32-bit" },
        { 0x40000001, 4, "overflow boundary" },

        /* Values that overflow 64-bit size_t multiplication */
        { (size_t)0xFFFFFFFFFFFFFFFFULL, 2, "max 64-bit x 2 - overflow" },
        { 2, (size_t)0xFFFFFFFFFFFFFFFFULL, "2 x max 64-bit - overflow" },
        { (size_t)0x8000000000000000ULL, 2, "half max 64-bit x 2 - overflow" },
        { (size_t)0x4000000000000001ULL, 4, "64-bit overflow boundary" },

        /* sizeof(double) multiplication overflow */
        { SIZE_MAX / sizeof(double) + 1, 1, "overflow with sizeof(double)" },
        { 1, SIZE_MAX / sizeof(double) + 1, "overflow with sizeof(double) reversed" },

        /* Zero dimensions */
        { 0, 100, "zero pointsNum" },
        { 100, 0, "zero elements" },
        { 0, 0, "both zero" },

        /* Large but potentially valid on 64-bit */
        { 1000000, 1, "1M x 1 - large but linear" },
        { 1, 1000000, "1 x 1M - large but linear" },
    };

    int num_cases = sizeof(adversarial_cases) / sizeof(adversarial_cases[0]);

    for (int i = 0; i < num_cases; i++) {
        size_t pointsNum = adversarial_cases[i].pointsNum;
        size_t elements = adversarial_cases[i].elements;

        /* Property 1: Safe multiplication must correctly detect overflow */
        size_t product;
        int mul_ok = safe_multiply_size(pointsNum, elements, &product);

        if (mul_ok && pointsNum > 0 && elements > 0) {
            /* Verify the multiplication result is correct */
            /* If no overflow, product / pointsNum should equal elements */
            ck_assert_msg(product / pointsNum == elements,
                "INVARIANT VIOLATED: multiplication result incorrect for case '%s': "
                "pointsNum=%zu, elements=%zu, product=%zu",
                adversarial_cases[i].description, pointsNum, elements, product);
        }

        /* Property 2: If overflow would occur, allocation must return NULL */
        double *buf = safe_alloc_matrix(pointsNum, elements);

        if (!mul_ok) {
            /* Overflow detected - allocation MUST return NULL */
            ck_assert_msg(buf == NULL,
                "INVARIANT VIOLATED: allocation succeeded despite overflow for case '%s': "
                "pointsNum=%zu, elements=%zu",
                adversarial_cases[i].description, pointsNum, elements);
        }

        /* Property 3: If allocation succeeded, buffer must be large enough */
        if (buf != NULL) {
            int bounds_ok = verify_buffer_bounds(buf, pointsNum, elements);
            ck_assert_msg(bounds_ok,
                "INVARIANT VIOLATED: buffer too small for writes for case '%s': "
                "pointsNum=%zu, elements=%zu",
                adversarial_cases[i].description, pointsNum, elements);
            free(buf);
        }
    }
}
END_TEST

START_TEST(test_integer_overflow_detection)
{
    /* Invariant: Overflow detection must work correctly for all boundary values */

    struct {
        size_t a;
        size_t b;
        int expect_overflow; /* 1 if overflow expected */
    } overflow_cases[] = {
        { 1, 1, 0 },
        { SIZE_MAX, 1, 0 },
        { 1, SIZE_MAX, 0 },
        { SIZE_MAX, 2, 1 },
        { 2, SIZE_MAX, 1 },
        { SIZE_MAX / 2 + 1, 2, 1 },
        { SIZE_MAX / 2, 2, 0 },
        { 0, SIZE_MAX, 0 },
        { SIZE_MAX, 0, 0 },
        { 0xFFFF, 0xFFFF, 0 },
        { 0x100000000ULL, 0x100000000ULL, 1 },
    };

    int num_cases = sizeof(overflow_cases) / sizeof(overflow_cases[0]);

    for (int i = 0; i < num_cases; i++) {
        size_t result;
        int ok = safe_multiply_size(overflow_cases[i].a, overflow_cases[i].b, &result);

        if (overflow_cases[i].expect_overflow) {
            ck_assert_msg(ok == 0,
                "INVARIANT VIOLATED: overflow not detected for a=%zu, b=%zu",
                overflow_cases[i].a, overflow_cases[i].b);
        } else {
            ck_assert_msg(ok == 1,
                "INVARIANT VIOLATED: false overflow detected for a=%zu, b=%zu",
                overflow_cases[i].a, overflow_cases[i].b);
            if (overflow_cases[i].a > 0 && overflow_cases[i].b > 0) {
                ck_assert_msg(result == overflow_cases[i].a * overflow_cases[i].b,
                    "INVARIANT VIOLATED: incorrect multiplication result for a=%zu, b=%zu",
                    overflow_cases[i].a, overflow_cases[i].b);
            }
        }
    }
}
END_TEST

START_TEST(test_allocation_size_consistency)
{
    /* Invariant: The allocated size must always be consistent with
     * the number of elements that will be written */

    size_t valid_sizes[][2] = {
        { 1, 1 },
        { 2, 2 },
        { 3, 4 },
        { 4, 5 },
        { 10, 10 },
        { 50, 50 },
        { 100, 4 },
        { 4, 100 },
        { 255, 255 },
        { 1024, 4 },
    };

    int num_cases = sizeof(valid_sizes) / sizeof(valid_sizes[0]);

    for (int i = 0; i < num_cases; i++) {
        size_t pointsNum = valid_sizes[i][0];
        size_t elements = valid_sizes[i][1];

        double *buf = safe_alloc_matrix(pointsNum, elements);

        /* For valid small sizes, allocation must succeed */
        ck_assert_msg(buf != NULL,
            "INVARIANT VIOLATED: allocation failed for valid size pointsNum=%zu, elements=%zu",
            pointsNum, elements);

        /* All elements must be writable without buffer overflow */
        size_t total = pointsNum * elements;
        for (size_t j = 0; j < total; j++) {
            buf[j] = 0.0;
        }

        /* Verify writes succeeded */
        for (size_t j = 0; j < total; j++) {
            ck_assert_msg(buf[j] == 0.0,
                "INVARIANT VIOLATED: write verification failed at index %zu", j);
        }

        free(buf);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security_MatrixAllocation");
    tc_core = tcase_create("Core");

    tcase_set_timeout(tc_core, 30);
    tcase_add_test(tc_core, test_matrix_allocation_no_overflow);
    tcase_add_test(tc_core, test_integer_overflow_detection);
    tcase_add_test(tc_core, test_allocation_size_consistency);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}