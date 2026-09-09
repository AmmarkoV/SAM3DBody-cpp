#ifndef BVH_TO_BVH_H_INCLUDED
#define BVH_TO_BVH_H_INCLUDED


#include "../bvh_loader.h"
#include "../calculate/bvh_transform.h"
#include "../mathLibrary.h"

#ifdef __cplusplus
extern "C"
{
#endif

int dumpBVHToBVH(
                  const char * bvhFilename,
                  struct BVH_MotionCapture * mc,
                  int writeHierarchy,
                  int writeMotion
                );

#ifdef __cplusplus
}
#endif

#endif // BVH_TO_BVH_H_INCLUDED
