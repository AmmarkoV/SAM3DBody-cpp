/** @file matrixTools.h
 *  @brief  Some common mathematic declarations , and GCC hints for loop taking
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIX_TOOLS_H_INCLUDED
#define MATRIX_TOOLS_H_INCLUDED

#define PRINT_MATRIX_DEBUGGING 0



#if __GNUC__
#define likely(x)    __builtin_expect (!!(x), 1)
#define unlikely(x)  __builtin_expect (!!(x), 0)
#else
 #define likely(x)   x
 #define unlikely(x)   x
#endif

#define PI 3.14159265358979323846264338

#define DEG2RAD 3.141592653589793f/180

// Pre-calculated value of PI / 180.
#define kPI180   0.017453

// Pre-calculated value of 180 / PI.
#define k180PI  57.295780

// Converts degrees to radians.
#define degreesToRadians(x) (x * kPI180)

// Converts radians to degrees.
#define radiansToDegrees(x) (x * k180PI)

#endif
