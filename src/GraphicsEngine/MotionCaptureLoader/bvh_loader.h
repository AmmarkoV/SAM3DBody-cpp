/** @file bvh_loader.h
 *  @brief  BVH file parser
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
            This is a very clean-cut parser of bvh file that will load a struct BVH_MotionCapture using the loadBVH command
            This part of the code does not perform any transformations on the loaded file, if you want to check transformations see bvh_transform.h
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef BVH_LOADER_H_INCLUDED
#define BVH_LOADER_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

//BVH Code version
static const char BVH_LOADER_VERSION_STRING [] = "0.74";

/**
* @brief Mathematical operations and structures such as the struct Matrix4x4OfFloats are declared here
* @ingroup BVH
*/
#include "mathLibrary.h"

/**
* @brief MAX_BVH_JOINT_NAME is the maximum label size for Joint names
* @ingroup BVH
*/
#define MAX_BVH_JOINT_NAME 32

/**
* @brief MAX_BVH_JOINT_HIERARCHY_SIZE is the maximum number of Joints
* The human file I use has 1 Root + 164 Joints + 47 End Sites = 212 I set a similar maximum here to ensure better caching..
* @ingroup BVH
*/
#define MAX_BVH_JOINT_HIERARCHY_SIZE 1550


/**
* @brief MAX_BVH_FILE_LINE_SIZE is the maximum number of a valid line in an incoming  BVH file
* @ingroup BVH
*/
#define MAX_BVH_FILE_LINE_SIZE 10000



static const char * channelNames[] =
{
    "no rotation channel",
    "Xposition",
    "Yposition",
    "Zposition",
    "Wrotation", //QBVH
    "Xrotation",
    "Yrotation",
    "Zrotation",
    "RodriguesX",
    "RodriguesY",
    "RodriguesZ",
//=================
    "End of Channel Names" ,
    "Unknown"
};


enum CHANNEL_NAMES
{
  BVH_CHANNEL_NONE=0,
  BVH_POSITION_X, //1
  BVH_POSITION_Y, //2
  BVH_POSITION_Z, //3
  BVH_ROTATION_W, //4 QBVH
  BVH_ROTATION_X, //5
  BVH_ROTATION_Y, //6
  BVH_ROTATION_Z, //7
  BVH_RODRIGUES_X,//8
  BVH_RODRIGUES_Y,//9
  BVH_RODRIGUES_Z,//10
  //---------------------
  BVH_VALID_CHANNEL_NAMES,
  //---------------------
  BVH_INVALID_CHANNEL
};



static const char * rotationOrderNames[] =
{
  "no rotation order",
  "XYZ",
  "XZY",
  "YXZ",
  "YZX",
  "ZXY",
  "ZYX",
  "QXQYQZQW", //QBVH
  "RODRIGUES",//QBVH
//=================
    "End of Channel Rotation Orders" ,
    "Unknown"
};


enum CHANNEL_ROTATION_ORDER
{
  BVH_ROTATION_ORDER_NONE=0,
  BVH_ROTATION_ORDER_XYZ,
  BVH_ROTATION_ORDER_XZY,
  BVH_ROTATION_ORDER_YXZ,
  BVH_ROTATION_ORDER_YZX,
  BVH_ROTATION_ORDER_ZXY,
  BVH_ROTATION_ORDER_ZYX,
  BVH_ROTATION_ORDER_QWQXQYQZ, //QBVH
  BVH_ROTATION_ORDER_RODRIGUES, //QBVH
  //--------------------
  BVH_VALID_ROTATION_ORDER_NAMES
};


/**
 * @brief Regardless of the rotation order above when transcations happen motion buffers
 * get their channels rerouted to constant positions so that it is easier to manage them!
 */
enum MOTIONBUFFER_TRANSACTION_DATA_FIELDS
{
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X=0,
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y,
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z,
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W, //QBVH
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X,
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y,
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z,
 //-----------------------------
 MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER
};


/**
 * @brief This is a relatively clean way to make sure all motion transactions adhere to the above fields..
 */
struct motionTransactionData
{
  float data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER];
};


/** * @brief Typedefs to make sure there are no logic errors when accessing a Joint */
typedef unsigned int BVHJointID;
/** * @brief Typedefs to make sure there are no logic errors when accessing a Frame*/
typedef unsigned int BVHFrameID;
/** * @brief Typedefs to make sure there are no logic errors when accessing a Motion Channel*/
typedef unsigned int BVHMotionChannelID;

/**
  * @brief MotionBuffers are a C "vector" container to flexibly carry, copy and process motions in different functions
  */
struct MotionBuffer
{
  float * motion;
  unsigned int bufferSize;
};


struct jointPartOfKnownHierarchy
{
   char isAPartOfLeftFoot;
   char isAPartOfRightFoot;
   char isAPartOfLeftArm;
   char isAPartOfRightArm;
   char isAPartOfLeftHand;
   char isAPartOfRightHand;
   char isAPartOfHead;
   char isAPartOfTorso;
};


/**
* @brief Each BVH Joint has a number of properties,offsets and channels that are modeled using this structure
* @ingroup BVH
*/
struct BVH_Joint
{
  //-------------------------------------------
  char isImmuneToTorsoOcclusions;
  //-------------------------------------------
  struct jointPartOfKnownHierarchy partOfHierarchy;
  //-------------------------------------------
  char erase2DCoordinates;
  char isRoot;
  char hasBrace;
  char isEndSite;
  char hasEndSite;
  //-------------------------------------------
  char jointName[MAX_BVH_JOINT_NAME+1];          //+Null terminator space
  char jointNameLowercase[MAX_BVH_JOINT_NAME+1]; //+Null terminator space
  unsigned int jointNameHash;
  //-------------------------------------------
  BVHJointID parentJoint;
  BVHJointID endSiteJoint;
  unsigned int hierarchyLevel;
  //-------------------------------------------
  float offset[3];
  struct Matrix4x4OfFloats staticTransformation;
  //-------------------------------------------
  char  hasPositionalChannels;
  char  hasRotationalChannels;
  char  hasQuaternionRotation;
  char  hasRodriguesRotation;
  char  channelType[BVH_VALID_CHANNEL_NAMES];
  char  resolveChannelType[BVH_VALID_CHANNEL_NAMES];
  char  channelRotationOrder;
  unsigned int loadedChannels;
  //-------------------------------------------
  //This is new functionality to define symmetries and flip L/R joints/poses
  BVHJointID symmetricJoint;
  unsigned int symmetryType;
  char symmetryIsLeftJoint;
  //-------------------------------------------
};



/**
* @brief The lookup table has 1:1 correspondance from joints to the given motions
* @ingroup BVH
*/
struct BVH_JointToMotion_LookupTable
{
  BVHMotionChannelID jointMotionOffset;
  BVHMotionChannelID channelIDMotionOffset[BVH_VALID_CHANNEL_NAMES]; //<- THIS IS BUGGED?
};

/**
* @brief The lookup table has 1:1 correspondance from motions to the given joints
* @ingroup BVH
*/
struct BVH_MotionToJoint_LookupTable
{
  BVHJointID jointID;
  BVHJointID parentID;
  BVHMotionChannelID channelID;
};


/**
* @brief Each BVH Motion Capture file has a hierarchy of BVH_Joints and an array of motion values that should be accessed using
         the helper functions
* @ingroup BVH
*/
struct BVH_MotionCapture
{
  float scaleWorld;

  // Header
  unsigned int numberOfFrames;
  unsigned int numberOfFramesEncountered;
  float frameTime;
  //--------

  //Joint Hierarchy
  unsigned int rootJointID;
  unsigned int MAX_jointHierarchySize;
  unsigned int jointHierarchySize;
  struct BVH_Joint * jointHierarchy; //End Site Joints need extra space so be sure they are accounted for in MAX_BVH_JOINT_HIERARCHY_SIZE
  //--------------

  //We may want to only work with specific selected joints..!
  unsigned int selectionIncludesEndSites;
  unsigned int numberOfJointsWeWantToSelect;
  unsigned int * selectedJoints;
  unsigned int * hideSelectedJoints;
  //----------------------------------------------------------

  //Lookup Tables..
  struct BVH_JointToMotion_LookupTable * jointToMotionLookup;
  struct BVH_MotionToJoint_LookupTable * motionToJointLookup;
  //---------------------------------------------------------------------------------------------

  //Motion
  unsigned int numberOfValuesPerFrame;
  unsigned int motionValuesSize;
  float * motionValues;
  //-------------------

  //Internal Variables..
  char * fileName;
  unsigned int linesParsed;
  //------------------------

  //---------
  char checkedForSymmetricJoints;
  char debug;
};


int enumerateRotationChannelOrderFromTypes(char typeA,char typeB,char typeC);

int enumerateChannelOrder(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint);


unsigned int bvh_resolveFrameAndJointAndChannelToMotionID(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID);

/**
* @brief Load a BVH file by giving a filename and filling in a BVH_MotionCapture struct
* @ingroup BVH
* @param  C-String with path to BVH File
* @param  pointer to an allocated BVH_MotionCapture struct
* @param  the scale of the world
*/
int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion, float scaleWorld);


/**
* @brief Free the space allocated by a BVH_MotionCapture file
* @ingroup BVH
* @param  BVH Structure
* @return 1=Success/0=Failure
*/
int bvh_free(struct BVH_MotionCapture * bvhMotion);


int bvh_setMIDValue(
               struct BVH_MotionCapture * mc,
               unsigned int mID,
               float value
              );

int bvh_SetPositionRotation(
                            struct BVH_MotionCapture * mc,
                            struct motionTransactionData * positionAndRotation
                           );


int bvh_OffsetPositionRotation(
                               struct BVH_MotionCapture * mc,
                               struct motionTransactionData * positionAndRotation
                              );


/**
* @brief Get Parent joint .
* @ingroup BVH
* @param  BVH Structure
* @param  Joint ID we want to query
* @return 1=HasParent/0=NoParent-Error
*/
int bvh_getJointParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID );


/**
* @brief Get Parent joint .
* @ingroup BVH
* @param  BVH Structure
* @param  Parent joint ID we want to make sure that the next supplied joint is a children of
* @param  Child of joint ID we want to make sure that is a children ( not necessarily the first one )
* @return 1=HasParent/0=NoParent-Error
*/
int bvh_isJointAChildrenID(
                           struct BVH_MotionCapture * bvhMotion,
                           BVHJointID parentJID,
                           BVHJointID childJID
                          );


int bvh_onlyAnimateGivenJoints(struct BVH_MotionCapture * bvhMotion,unsigned int numberOfArguments,const char **argv);

/**
* @brief Ask if joint has a parent.
* @ingroup BVH
* @param  BVH Structure
* @param  Joint ID we want to query
* @return 1=HasParent/0=NoParent-Error
*/
int bvh_jointHasParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID );





int bvh_jointGetEndSiteChild(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,BVHJointID * jChildID);


/**
* @brief Ask if joint has rotational component.
* @ingroup BVH
* @param  BVH Structure
* @param  Joint ID we want to query
* @return 1=HasRotation/0=NoRotation-Error
*/
int bvh_jointHasRotation(struct BVH_MotionCapture * bvhMotion , BVHJointID jID);

/**
* @brief Resolve a C-String from a Joint name to a Joint ID
* @ingroup BVH
* @param  BVH Structure
* @param  C-String with Joint Name
* @param  Output Joint ID
* @return 1=Found/0=NotFound
*/
int bvh_getJointIDFromJointName( struct BVH_MotionCapture * bvhMotion , const char * jointName, BVHJointID * jID);

int bvh_getJointIDFromJointNameNocase(
                                      struct BVH_MotionCapture * bvhMotion ,
                                      const char * jointName,
                                      BVHJointID * jID
                                     );

/**
* @brief Get the root joint of a BVH_MotionCapture file
* @ingroup BVH
* @param   BVH Structure
* @param   Output Root Joint ID
* @return  1=Found/0=NotFound
*/
int bvh_getRootJointID(
                       struct BVH_MotionCapture * bvhMotion ,
                       BVHJointID * jID
                      );



/**
* @brief Copy a motion frame of a BVH_MotionCapture file
* @ingroup BVH
* @param  BVH Structure
* @param  Target frame we want to populate with source
* @param  Source frame we want to copy
* @return 1=Success/0=Failure
*/
int bvh_copyMotionFrame(
                         struct BVH_MotionCapture * bvhMotion,
                         BVHFrameID tofID,
                         BVHFrameID fromfID
                        );


/**
* @brief Copy a motion frame to a buffer
* @ingroup BVH
* @param  BVH Structure
* @param  Target frame we want to populate with source
* @param  Source frame we want to copy
* @return 1=Success/0=Failure
*/
int bvh_copyMotionFrameToMotionBuffer(
                                      struct BVH_MotionCapture * bvhMotion,
                                      struct MotionBuffer * motionBuffer,
                                      BVHFrameID fromfID
                                     );


/**
* @brief Copy a motion buffer to a motion frame
* @ingroup BVH
* @param  BVH Structure
* @param  Target frame we want to copy to
* @param  Source motion buffer we want to copy from
* @return 1=Success/0=Failure
*/
int bvh_copyMotionBufferToMotionFrame(
                                       struct BVH_MotionCapture * bvhMotion,
                                       BVHFrameID fromfID,
                                       struct MotionBuffer * motionBuffer
                                     );

//float * bvh_getJointOffset(struct BVH_MotionCapture * bvhMotion , BVHJointID jID);


int bvh_getJointDimensions(
                              struct BVH_MotionCapture * bvhMotion,
                              const char * jointName,
                              float * xValue,
                              float * yValue,
                              float * zValue
                             );

int bvh_changeJointDimensions(
                              struct BVH_MotionCapture * bvhMotion,
                              const char * jointName,
                              float xScale,
                              float yScale,
                              float zScale
                             );


int bvh_scaleAllOffsets(
                        struct BVH_MotionCapture * bvhMotion,
                        float scalingRatio
                       );



int bvh_scaleAllJointChildrenOffsets(
                                     struct BVH_MotionCapture * bvhMotion,
                                     const char * jointName,
                                     float scalingRatio
                                    );


int bvh_getMotionChannelName(struct BVH_MotionCapture * bvhMotion,BVHMotionChannelID mID,char * target,unsigned int targetLength);

/**
* @brief Request a specific motion channel of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @param  Type of channel we want to access ( see enum CHANNEL_NAMES )
* @return Value of channel for specific joint at specific frame
*/
float bvh_getJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID);



/**
* @brief Request X Rotation of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return X Rotation for specific joint at specific frame
*/
float  bvh_getJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Y Rotation of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Y Rotation for specific joint at specific frame
*/
float  bvh_getJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Z Rotation of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Z Rotation for specific joint at specific frame
*/
float  bvh_getJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request X Position of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return X Position for specific joint at specific frame
*/
float  bvh_getJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Y Position of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Y Position for specific joint at specific frame
*/
float  bvh_getJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Z Position of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Z Position for specific joint at specific frame
*/
float  bvh_getJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);




//TODO: add doxygen documentation
int bvh_setJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID,float value);

int bvh_setJointRotationWAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);
int bvh_setJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);
int bvh_setJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);
int bvh_setJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);

int bvh_setJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);
int bvh_setJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);
int bvh_setJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value);

int bvh_setPosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData);



/**
* @brief Request XYZ positions and XYZ rotations that will be written on a float data[6] array
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @param  Output float array that will hold the results
* @param  Size of the output float array that should hold at least 6 floats
* @return 1=Success/0=Failure
*/
int bvh_populatePosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData);


float  bvh_getJointRotationWAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);
float  bvh_getJointRotationXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);
float  bvh_getJointRotationYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);
float  bvh_getJointRotationZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);
float  bvh_getJointPositionXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);
float  bvh_getJointPositionYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);
float  bvh_getJointPositionZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer);

/**
* @brief A very important call that handles retreiving data from the motion buffer..
* The data output array should be at least MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER elements long
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @param  Output float array that will hold the results
* @param  Size of the output float array that should hold at least 6 floats
* @return 1=Success/0=Failure
*/
int bvh_retrieveDataFromMotionBuffer(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , float * motionBuffer, float * data, unsigned int sizeOfData);

/**
* @brief Direct access to the motion data, without Joint hierarchy,Frame separation etc, should not be used unless you really know what you are doing..
* @ingroup BVH
* @param  BVH Structure
* @param  Motion element we want to read
* @return Motion Value, 0.0 on failure
*/
float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion ,BVHMotionChannelID mID);


/**
* @brief Direct access to the motion data, without Joint hierarchy,Frame separation etc, should not be used unless you really know what you are doing..
* @ingroup BVH
* @param  BVH Structure
* @param  Motion element we want to write
* @param  Motion ID to change
* @return 1=Success/0=Failure
*/
int bvh_setMotionValue(struct BVH_MotionCapture * bvhMotion,BVHMotionChannelID mID,float * value);

/**
* @brief Direct access to the motion data, without Joint hierarchy for a particular frame, should not be used unless you really know what you are doing..
* @ingroup BVH
* @param  BVH Structure
* @param  Frame we want to access
* @param  Motion element we want to read
* @return Motion Value, 0.0 on failure
*/
float bvh_getMotionValueOfFrame(struct BVH_MotionCapture * bvhMotion,BVHFrameID fID,BVHMotionChannelID mID);

/**
* @brief Direct access to the motion data, without Joint hierarchy for a particular frame, should not be used unless you really know what you are doing..
* @ingroup BVH
* @param  BVH Structure
* @param  Frame we want to access
* @param  Motion ID to change
* @return 1=Success/0=Failure
*/
int bvh_setMotionValueOfFrame(struct BVH_MotionCapture * bvhMotion,BVHFrameID fID,BVHMotionChannelID mID,float * value);

/**
* @brief Perform a select query marking all the children of a specific joint ..
* @ingroup BVH
* @param  BVH Structure
* @param  Joint name we want to select plus all its children
* @return 1=Success/0=Failure
*/
int bvh_selectChildrenOfJoint(struct BVH_MotionCapture * mc, const char * parentJoint);


void bvh_considerIfJointIsSelected(
                                   struct BVH_MotionCapture * mc,
                                   unsigned int jID,
                                   int * isJointSelected,
                                   int * isJointEndSiteSelected
                                  );

int bvh_selectJoints(
                    struct BVH_MotionCapture * mc,
                    unsigned int numberOfValues,
                    unsigned int includeEndSites,
                    const char **argv,
                    unsigned int iplus1
                   );



int bvh_selectJointsToHide2D(
                             struct BVH_MotionCapture * mc,
                             unsigned int numberOfValues,
                             unsigned int includeEndSites,
                             const char **argv,
                             unsigned int iplus1
                            );



int bvh_removeSelectedFrames(struct BVH_MotionCapture * bvhMotion,unsigned int * framesToRemove);




void freeMotionBuffer(struct MotionBuffer ** pointerToMB);
int copyMotionBuffer(struct MotionBuffer * dst,struct MotionBuffer * src);
struct MotionBuffer * mallocNewMotionBuffer(struct BVH_MotionCapture * mc);
struct MotionBuffer * mallocNewMotionBufferAndCopy(struct BVH_MotionCapture * mc,struct MotionBuffer * whatToCopy);
void compareMotionBuffers(const char * msg,struct MotionBuffer * guess,struct MotionBuffer * groundTruth);
void compareTwoMotionBuffers(struct BVH_MotionCapture * mc,const char * msg,struct MotionBuffer * guessA,struct MotionBuffer * guessB,struct MotionBuffer * groundTruth);


/**
* @brief Print BVH information on stderr
* @ingroup BVH
* @param  pointer to an allocated BVH_MotionCapture struct
*/
void bvh_printBVH(struct BVH_MotionCapture * bvhMotion);

void bvh_printBVHJointToMotionLookupTable(struct BVH_MotionCapture * bvhMotion);




#ifdef __cplusplus
}
#endif




#endif // BVH_LOADER_H_INCLUDED
