#ifndef BVH_IMPORTFROMBVH_H_INCLUDED
#define BVH_IMPORTFROMBVH_H_INCLUDED

#include "../bvh_loader.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif


void bvh_populateStaticTransformationOfJoint(struct BVH_MotionCapture * bvhMotion,BVHJointID jID);

int readBVHHeader(struct BVH_MotionCapture * bvhMotion , FILE * fd );


int readBVHMotion(struct BVH_MotionCapture * bvhMotion , FILE * fd );

#ifdef __cplusplus
}
#endif

#endif //BVH_IMPORTFROMBVH_H_INCLUDED
