#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#include <stdint.h>
struct timespec {
    long tv_sec;
    long tv_nsec;
};
#define CLOCK_MONOTONIC 0
static int clock_gettime(int ignore, struct timespec* tv) {
    uint64_t ns = (uint64_t)GetTickCount64() * 1000000;
    tv->tv_sec = (long)(ns / 1000000000);
    tv->tv_nsec = (long)(ns % 1000000000);
    return 0;
}
#else
#include <time.h>
#endif

#include "../matrix4x4Tools.h"

unsigned long tickBaseMN=0;

unsigned long GetTickCountMicrosecondsMN()
{
    struct timespec ts;
    if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0)
    {
        return 0;
    }

    if (tickBaseMN==0)
    {
        tickBaseMN = ts.tv_sec*1000000 + ts.tv_nsec/1000;
        return 0;
    }

    return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBaseMN;
}


int main()
{

    if (!codeHasSSE())
    {
      printf("SSE code is not compiled in..\n");
      exit(0);
    }

    int i=0;

    struct Matrix4x4OfFloats testResultOptimized={0};
    struct Matrix4x4OfFloats testResultUnoptimized={0};
    struct Matrix4x4OfFloats matrixA={0};
    struct Matrix4x4OfFloats matrixB={0};

    //Set matrices to identity
    matrixA.m[0]=1.0; matrixA.m[5]=1.0; matrixA.m[10]=1.0; matrixA.m[15]=1.0;
    matrixB.m[0]=1.0; matrixB.m[5]=1.0; matrixB.m[10]=1.0; matrixB.m[15]=1.0;

    unsigned int numberOfSamples = 100000;
    unsigned long unoptimizedTime = 0;
    unsigned long optimizedTime = 0;

    unsigned int errors = 0;
    for (i=0; i<numberOfSamples; i++)
    {
        float tmp = rand()%1000 / 100;
        matrixA.m[1] = tmp;
        matrixB.m[1] = tmp;

        unsigned long startUnoptimized = GetTickCountMicrosecondsMN();
        //multiplyTwo4x4FMatrices_Naive(testResultUnoptimized.m,matrixA.m,matrixB.m);
        unsigned long endUnoptimized = GetTickCountMicrosecondsMN();
        unoptimizedTime+=endUnoptimized - startUnoptimized;

        unsigned long startOptimized = GetTickCountMicrosecondsMN();
        #if INTEL_OPTIMIZATIONS
        multiplyTwoRaw4x4FMatricesS(testResultOptimized.m,matrixA.m,matrixB.m);
        #endif
        unsigned long endOptimized = GetTickCountMicrosecondsMN();
        optimizedTime+=endOptimized - startOptimized;

        if (matrixA.m[1]!=matrixB.m[1])
        {
            ++errors;
        }
    }

    if (errors>0)
    {
        fprintf(stderr,"%u errors encountered..\n",errors);
    }

    print4x4FMatrix("Unoptimized Result",testResultUnoptimized.m,1);
    print4x4FMatrix("Optimized Result",testResultOptimized.m,1);


    printf("Finished with %u samples !\n",numberOfSamples);
    printf("%0.4f microseconds unoptimized!\n",(float) unoptimizedTime/numberOfSamples);
    printf("%0.4f microseconds optimized!\n",(float) optimizedTime/numberOfSamples);
    return 0;
}
