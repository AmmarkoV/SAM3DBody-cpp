#ifndef BVH_TO_TRI_POSE_H_INCLUDED
#define BVH_TO_TRI_POSE_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif


struct BVH_RotationOrder
{
  char label[64];
  float sign;
  char  rotID;
};

struct BVH_JointAssocation
{
  char bvhJointName[MAX_BVH_JOINT_NAME];
  char triJointName[MAX_BVH_JOINT_NAME];
  int useJoint;

  struct BVH_RotationOrder rotationOrder[3];

  float offset[3];
};

struct bvhToTRI
{
 unsigned int numberOfJointAssociations;
 struct BVH_JointAssocation jointAssociation[MAX_BVH_JOINT_HIERARCHY_SIZE];
};


int bvh_loadBVHToTRIAssociationFile(
                                    const char * filename ,
                                    struct bvhToTRI * bvhtri,
                                    struct BVH_MotionCapture * mc
                                  );


#ifdef __cplusplus
}
#endif

#endif // BVH_TO_TRI_POSE_H_INCLUDED
