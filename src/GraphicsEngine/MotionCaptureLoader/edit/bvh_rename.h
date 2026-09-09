#ifndef BVH_RENAME_H_INCLUDED
#define BVH_RENAME_H_INCLUDED


#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

void lowercase(char *str);

void uppercase(char *str);


unsigned long hashFunctionJoints(const char *str);

void bvh_updateJointNameHashes(struct BVH_MotionCapture * bvhMotion);



/**
* @brief Force all joint names to be lowercase
* @ingroup BVH
* @param  BVH Structure
*/
void bvh_renameJointsToLowercase(struct BVH_MotionCapture * bvhMotion);


/**
* @brief Different motion capture systems produce different types of joint names. For example lhip can be named lefthip,leftupleg,lthigh,leftupperLeg etc.
*        This call renames them in order to ensure better compatibility with various different motion capture files..
* @ingroup BVH
* @param  BVH Structure
*/
void bvh_renameJointsForCompatibility(struct BVH_MotionCapture * bvhMotion);


#ifdef __cplusplus
}
#endif



#endif
