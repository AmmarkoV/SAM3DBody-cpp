/** @file bvh_transform.h
 *  @brief  BVH 3D Transormation code.
 *          This part of the code does not deal with loading or changing the BVH file but just performing 3D transformations.
 *          In order to keep things clean all transforms take place in the BVH_Transform structure. The 3D calculation code is
 *          also seperated using the AmMatrix sublibrary https://github.com/AmmarkoV/RGBDAcquisition/tree/master/tools/AmMatrix
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef BVH_TRANSFORM_H_INCLUDED
#define BVH_TRANSFORM_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif





//Hashing means to generate a new table that only contains
//The needed fields for transforms in order to try skipping
//checks we know that will fail
#define USE_TRANSFORM_HASHING 0

#if USE_TRANSFORM_HASHING
 #warning "Transform hashing is under construction and not yet working.."
#endif


//Transforms should happen on dynamically allocated memory blocks..
#define DYNAMIC_TRANSFORM_ALLOCATIONS 0


//MAX_BVH_JOINT_HIERARCHY_SIZE
#define MAX_BVH_TRANSFORM_SIZE MAX_BVH_JOINT_HIERARCHY_SIZE





/*
 4x4 Matrix Order
   0  1  2  3
   4  5  6  7
   8  9 10 11
  12 13 14 15
*/
//--------------------------------
//--------------------------------
//--------------------------------
struct rectangle3DPointsArea
{
  float x1,y1,z1;
  float x2,y2,z2;
  float x3,y3,z3;
  float x4,y4,z4;
};
//--------------------------------
struct rectangle2DPointsArea
{
  char calculated;
  float x1,y1;
  float x2,y2;
  float x3,y3;
  float x4,y4;

  float x,y,width,height;
};
//--------------------------------
struct rectangleArea
{
  char exists,point1Exists,point2Exists,point3Exists,point4Exists;
  int jID[4];
  float averageDepth;
  struct rectangle2DPointsArea rectangle2D;
  struct rectangle3DPointsArea rectangle3D;
};
//--------------------------------
//--------------------------------
//--------------------------------
struct triangle3DPointsArea
{
  float x1,y1,z1;
  float x2,y2,z2;
  float x3,y3,z3;
};
//--------------------------------
struct triangle2DPointsArea
{
  char calculated;
  float x1,y1;
  float x2,y2;
  float x3,y3;

  float x,y,width,height;
};
//--------------------------------
struct triangleArea
{
  char exists,point1Exists,point2Exists,point3Exists;
  int jID[3];
  float averageDepth;
  struct triangle2DPointsArea triangle2D;
  struct triangle3DPointsArea triangle3D;
};
//--------------------------------
//--------------------------------
//--------------------------------

/**
    @struct BVH_TransformedJoint
    @brief Structure to store the transformed joint information.
    This structure stores the transformed joint information including flags, transforms, and positions.
    @var BVH_TransformedJoint::pos2DCalculated
    Flag indicating if the position in 2D has been calculated.
    @var BVH_TransformedJoint::isBehindCamera
    Flag indicating if the joint is behind the camera.
    @var BVH_TransformedJoint::isOccluded
    Flag indicating if the joint is occluded.
    @var BVH_TransformedJoint::isChainTransformationComputed
    Flag indicating if the chain transformation has been computed.
    @var BVH_TransformedJoint::localToWorldTransformation
    The local to world transformation matrix.
    @var BVH_TransformedJoint::chainTransformation
    The chain transformation matrix.
    @var BVH_TransformedJoint::dynamicTranslation
    The dynamic translation matrix.
    @var BVH_TransformedJoint::dynamicRotation
    The dynamic rotation matrix.
    @var BVH_TransformedJoint::pos3D
    Position of the joint in 3D space as X,Y,Z.
    @var BVH_TransformedJoint::pos2D
    Position of the joint in 2D space as X,Y.
    */
struct BVH_TransformedJoint
{
  //Flags
  //-----------------
  char pos2DCalculated;
  char isBehindCamera;
  char isOccluded;
  //char skipCalculations;
  char isChainTransformationComputed;

  //Transforms
  //-----------------
  struct Matrix4x4OfFloats localToWorldTransformation;
  struct Matrix4x4OfFloats chainTransformation;
  struct Matrix4x4OfFloats dynamicTranslation;
  struct Matrix4x4OfFloats dynamicRotation;

  //Position as X,Y,Z
  //-----------------
  float pos3D[4];

  //Position as 2D X,Y
  //-----------------
  float pos2D[2];
};

/**
    @struct BVH_Transform
    @brief This struct contains all the data related to transformation of a BVH motion capture structure.
    This struct contains the following fields:
        Memory Management flags:
        transformStructInitialized : A flag that indicates whether the struct has been initialized or not.
        numberOfJointsSpaceAllocated : The number of joints that have space allocated.
        numberOfJointsToTransform : The number of joints that need to be transformed.
        Skip Joint Optimization Logic:
        useOptimizations : A flag that indicates whether to use optimizations or not.
        skipCalculationsForJoint : An array that holds the information of which joints are skipped.
        Transform Hashing:
        jointIDTransformHashPopulated : A flag that indicates whether the joint ID transform hash is populated or not.
        lengthOfListOfJointIDsToTransform : The length of the list of joint IDs to be transformed.
        listOfJointIDsToTransform : An array that holds the IDs of the joints that need to be transformed.
        Actual Transformation data:
        torsoTriangle : A triangle area representing the torso.
        torso : A rectangle area representing the torso.
        joint : An array that holds the transformed joint data.
        centerPosition : The center position of the BVH motion capture.
        jointsOccludedIn2DProjection : The number of joints that are occluded in 2D projection.
        */
struct BVH_Transform
{
  //Memory Management flags..
  char transformStructInitialized;
  unsigned int numberOfJointsSpaceAllocated;
  unsigned int numberOfJointsToTransform;

  //Skip Joint Optimization Logic
  char useOptimizations;

  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   unsigned char * skipCalculationsForJoint;
  #else
   unsigned char skipCalculationsForJoint[MAX_BVH_TRANSFORM_SIZE];
  #endif

  //Transform hashing
  unsigned int jointIDTransformHashPopulated;
  unsigned int lengthOfListOfJointIDsToTransform;
  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   BVHJointID * listOfJointIDsToTransform;
  #else
   BVHJointID listOfJointIDsToTransform[MAX_BVH_TRANSFORM_SIZE];
  #endif

  //Actual Transformation data
  struct triangleArea torsoTriangle;
  struct rectangleArea torso;
  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   struct BVH_TransformedJoint * joint;
  #else
   struct BVH_TransformedJoint joint[MAX_BVH_TRANSFORM_SIZE];
  #endif
  float centerPosition[3];
  unsigned int jointsOccludedIn2DProjection;
};


/**
    @brief Calculates the distance of a joint from the torso plane in a BVH motion capture file.
    @param mc Pointer to a BVH_MotionCapture struct containing the motion capture data.
    @param bvhTransform Pointer to a BVH_Transform struct containing the transformed joint data.
    @param jID ID of the joint to calculate the distance for.
    @return The distance of the specified joint from the torso plane.
*/
float bvh_DistanceOfJointFromTorsoPlane(
                                        struct BVH_MotionCapture * mc,
                                        struct BVH_Transform * bvhTransform,
                                        BVHJointID jID
                                        );

/**
    @brief Populate a 2D rectangle from the projections of a joint in the BVH structure.
    @param mc The BVH Motion Capture structure
    @param bvhTransform The BVH Transform structure
    @param area The rectangleArea structure to be populated with the 2D projections
    @return Returns 1 if rectangle is calculated successfully, otherwise returns 0.
*/
int bvh_populateRectangle2DFromProjections(
                                           struct BVH_MotionCapture * mc ,
                                           struct BVH_Transform * bvhTransform,
                                           struct rectangleArea * area
                                          );

/**
    @brief Prints the information of the BVH Transform structure
    This function prints the information of the BVH Transform structure, including the size of the joint hierarchy, whether
    optimizations are used, whether the joint ID transform hash is populated, the length of the list of joint IDs to transform,
    the list of joint IDs to transform, the number of joints occluded in 2D projection, the center position of the joints,
    and information about each joint such as the name, whether the 2D position has been calculated, whether the joint is behind
    the camera, whether the joint is occluded, and whether the chain transformation has been calculated.
    @param label a string describing the label of the BVH Transform structure
    @param bvhMotion a pointer to the BVH Motion Capture structure
    @param bvhTransform a pointer to the BVH Transform structure
    */
void bvh_printBVHTransform(const char * label,struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform);

/**
    @brief Print the joints that are not skipped in the bvhTransform struct
    This function iterates over all the joints in the bvhMotion struct and checks if the joint is not skipped in the bvhTransform struct.
    If the joint is not skipped, it prints the joint ID and its name to the stderr.
    @param bvhMotion A pointer to the BVH_MotionCapture struct
    @param bvhTransform A pointer to the BVH_Transform struct
    @return void
    */
void bvh_printNotSkippedJoints(struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform);

/**
    @brief Check if a joint should be transformed given the current optimization settings
    @param bvhTransform The transform data structure
    @param jID The joint ID to check
    @return 1 if the joint should be transformed, 0 otherwise
*/
unsigned char bvh_shouldJointBeTransformedGivenOurOptimizations(const struct BVH_Transform * bvhTransform,const BVHJointID jID);

/**
    @brief Mark all joints in the BVH Motion Capture as useful in the BVH Transform
    This function sets the useOptimizations flag in the BVH Transform to 0, which means
    that all joints in the BVH Motion Capture will be used in the transform calculations.
    The skipCalculationsForJoint flag for each joint is also set to 0.
    If the USE_TRANSFORM_HASHING preprocessor flag is defined, the function bvh_HashUsefulJoints
    is called to update the listOfJointIDsToTransform.
    @param bvhMotion Pointer to the BVH Motion Capture data
    @param bvhTransform Pointer to the BVH Transform data
*/
int bvh_markAllJointsAsUsefullInTransform(
                                          struct BVH_MotionCapture * bvhMotion ,
                                          struct BVH_Transform * bvhTransform
                                         );

/**
    @brief Mark all joints as useless in transform
    This function marks all joints in the BVH hierarchy as useless in the transform structure. This means that calculations and transformations will not be performed on these joints.
    @param bvhMotion A pointer to the BVH motion capture structure
    @param bvhTransform A pointer to the BVH transform structure
    @return 1 if successful, 0 otherwise
    */
int bvh_markAllJointsAsUselessInTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform);


/**
    @brief Mark a joint and its parents as useful for transformation
    This function marks the given joint and all its parent joints as useful for transformation.
    The function is intended to be used with the bvh_loadTransformForMotionBuffer function.
    @param bvhMotion pointer to a BVH_MotionCapture struct
    @param bvhTransform pointer to a BVH_Transform struct
    @param jID the joint ID that we want to mark as useful for transformation
    @return int returns 1 if successful, otherwise 0
    */
int bvh_markJointAndParentsAsUsefulInTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform,BVHJointID jID);

/**
    @brief Mark a specific joint and its parents as useless in the transform process.
    This function is used to optimize the BVH transform process by marking specific joints and its parents as useless,
    so that they are not transformed. This can significantly improve performance when only a specific subset of the joints
    need to be transformed.
    @param bvhMotion A pointer to the BVH_MotionCapture struct that contains the joint hierarchy information.
    @param bvhTransform A pointer to the BVH_Transform struct that contains the transform information.
    @param jID The joint ID of the joint that needs to be marked as useful and its parents as useless.
    @return Returns 1 on success, 0 on failure.
    */
int bvh_markJointAsUsefulAndParentsAsUselessInTransform(
                                                        struct BVH_MotionCapture * bvhMotion ,
                                                        struct BVH_Transform * bvhTransform,
                                                        BVHJointID jID
                                                       );
/**
    @brief Mark a joint and its parents as useless in the BVH Transform
    This function will mark a joint and its parents in the BVH Transform as useless
    in order to skip the calculation of their transformations. This is useful when the
    joint or its parents are not needed for further processing.
    @param bvhMotion The BVH motion capture data
    @param bvhTransform The BVH Transform data
    @param jID The joint ID of the joint to be marked as useless
*/
int bvh_markJointAndParentsAsUselessInTransform(
                                                struct BVH_MotionCapture * bvhMotion ,
                                                struct BVH_Transform * bvhTransform,
                                                BVHJointID jID
                                              );


/**
    @brief This function loads a BVH motion capture data and applies the transformations specified in the motion buffer to a list of joints, as specified by the list of joint IDs.
    @param bvhMotion A pointer to the BVH motion capture data.
    @param motionBuffer A pointer to the motion buffer containing the motion data.
    @param bvhTransform A pointer to the BVH transform data.
    @param populateTorso A flag indicating whether to populate the torso information from the 3D transform.
    @param listOfJointIDsToTransform A pointer to an array of joint IDs specifying which joints should be transformed.
    @param lengthOfJointIDList The number of elements in the array of joint IDs.
*/
int bvh_loadTransformForMotionBufferFollowingAListOfJointIDs(
                                                             struct BVH_MotionCapture * bvhMotion,
                                                             float * motionBuffer,
                                                             struct BVH_Transform * bvhTransform,
                                                             unsigned int populateTorso,
                                                             BVHJointID * listOfJointIDsToTransform,
                                                             unsigned int lengthOfJointIDList
                                                            );

/**
    @brief Loads the transformed joint data for a motion buffer in a BVH motion capture file.
    @param bvhMotion Pointer to a BVH_MotionCapture struct containing the motion capture data.
    @param motionBuffer Pointer to a float buffer containing motion data for a specific frame.
    @param bvhTransform Pointer to a BVH_Transform struct to store the transformed joint data.
    @param populateTorso Flag indicating whether to calculate the torso triangle data.
    @return 0 on success, non-zero on failure.
    */
int bvh_loadTransformForMotionBuffer(
                                     struct BVH_MotionCapture * bvhMotion,
                                     float * motionBuffer,
                                     struct BVH_Transform * bvhTransform,
                                     unsigned int populateTorso
                                   );

/**
    @brief Loads the transformed joint data for a specific frame in a BVH motion capture file.
    @param bvhMotion Pointer to a BVH_MotionCapture struct containing the motion capture data.
    @param fID ID of the frame to load the transformed joint data for.
    @param bvhTransform Pointer to a BVH_Transform struct to store the transformed joint data.
    @param populateTorso Flag indicating whether to calculate the torso triangle data.
*/
int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion ,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform,
                               unsigned int populateTorso
                             );

/**
    @brief Removes the translation component from the transformed joint data of a BVH motion capture file.
    @param bvhMotion Pointer to a BVH_MotionCapture struct containing the motion capture data.
    @param bvhTransform Pointer to a BVH_Transform struct containing the transformed joint data.
    @bug  The function body has a fprintf statement that says "bvh_removeTranslationFromTransform not correctly implemented" which implies that the function is not working as expected and should be used with caution.
    @return 0 on success, non-zero on failure.
    */
int bvh_removeTranslationFromTransform(
                                       struct BVH_MotionCapture * bvhMotion ,
                                       struct BVH_Transform * bvhTransform
                                      );


/**
    @brief Allocate memory for the BVH transform structure
    This function is responsible for allocating memory for the BVH transform structure, it will check if the memory is already allocated,
    if it is it will only change the number of joints that need to be transformed, if not it will free the existing memory and allocate
    new memory for the transform structure.
    @param bvhMotion pointer to the BVH_MotionCapture structure
    @param bvhTransform pointer to the BVH_Transform structure
    @return int returns 1 if the allocation is successful, 0 otherwise
    @note This function uses dynamic allocation, if DYNAMIC_TRANSFORM_ALLOCATIONS is not defined the function will always return 0.
    */
int bvh_allocateTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform);


/**
    @brief This function frees the memory allocated for a BVH_Transform struct
    This function frees the memory allocated for a BVH_Transform struct
    @param bvhTransform pointer to the BVH_Transform struct
    @return returns 1 if successful, 0 if bvhTransform is null
    */
int bvh_freeTransform(struct BVH_Transform * bvhTransform);

#ifdef __cplusplus
}
#endif



#endif // BVH_TRANSFORM_H_INCLUDED
