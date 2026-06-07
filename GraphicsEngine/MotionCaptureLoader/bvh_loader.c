/*
    Written by Ammar Qammaz a.k.a. AmmarkoV 2018
    --
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <math.h>
#include "bvh_loader.h"
#include "edit/bvh_rename.h"
#include "../TrajectoryParser/InputParser_C.h"

#include "edit/bvh_cut_paste.h"

#include "import/fromBVH.h"


//----------------------------------------------------------
#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
//----------------------------------------------------------
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
//----------------------------------------------------------





//A very brief documentation of the BVH spec :
//http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        HIERARCHY PARSING
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------


int enumerateRotationChannelOrderFromTypes(char typeA,char typeB,char typeC)
{
  if ( (typeA==BVH_RODRIGUES_X) && (typeB==BVH_RODRIGUES_Y) && (typeC==BVH_RODRIGUES_Z) )
          { return BVH_ROTATION_ORDER_RODRIGUES; }

  switch (typeA)
  {
    case BVH_ROTATION_X :
         switch (typeB)
         {
           case BVH_ROTATION_Y :
               if (typeC == BVH_ROTATION_Z) { return BVH_ROTATION_ORDER_XYZ; }
           break;
           case BVH_ROTATION_Z :
               if (typeC == BVH_ROTATION_Y) { return BVH_ROTATION_ORDER_XZY; }
           break;
         };
    break;

    case BVH_ROTATION_Y :
         switch (typeB)
         {
           case BVH_ROTATION_X :
               if (typeC == BVH_ROTATION_Z) { return BVH_ROTATION_ORDER_YXZ; }
           break;
           case BVH_ROTATION_Z :
               if (typeC == BVH_ROTATION_X) { return BVH_ROTATION_ORDER_YZX; }
           break;
         };
    break;

    case BVH_ROTATION_Z :
         switch (typeB)
         {
           case BVH_ROTATION_X :
               if (typeC == BVH_ROTATION_Y) { return BVH_ROTATION_ORDER_ZXY; }
           break;
           case BVH_ROTATION_Y :
               if (typeC == BVH_ROTATION_X) { return BVH_ROTATION_ORDER_ZYX; }
           break;
         };
    break;
  }
 return BVH_ROTATION_ORDER_NONE;
}


int enumerateRotationChannelOrderFromTypePositions(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint,int posA,int posB,int posC)
{
  int channelOrder=enumerateRotationChannelOrderFromTypes(
                                                           bvhMotion->jointHierarchy[currentJoint].channelType[posA],
                                                           bvhMotion->jointHierarchy[currentJoint].channelType[posB],
                                                           bvhMotion->jointHierarchy[currentJoint].channelType[posC]
                                                         );
  return channelOrder;
}


#define NEW_CHANNELROTATION_ENUMERATION_CODE 1


#if NEW_CHANNELROTATION_ENUMERATION_CODE
int enumerateChannelOrder(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint)
{
  int channelOrder=BVH_ROTATION_ORDER_NONE;
  //-----------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------
  bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation = 0;
  bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels = 0;
  bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels = 0;
  bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation  = 0;
  //-----------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------
  for (unsigned int i=0; i<bvhMotion->jointHierarchy[currentJoint].loadedChannels; i++)
  {
    if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_POSITION_X) ||
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_POSITION_Y) ||
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_POSITION_Z)
       )
    {  bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=1; }
    //-----------------------------------------------------------------------------------
    if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_ROTATION_X) ||
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_ROTATION_Y) ||
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_ROTATION_Z)
       )
    { bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;  }
    //-----------------------------------------------------------------------------------
    if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_RODRIGUES_X) ||
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_RODRIGUES_Y) ||
          (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_RODRIGUES_Z)
       )
    {
      bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
      bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=1;
    }
    //-----------------------------------------------------------------------------------
    if (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_ROTATION_W)
    { bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=1;  }
    //-----------------------------------------------------------------------------------
  }
 //-----------------------------------------------------------------------------------
 //-----------------------------------------------------------------------------------
 //-----------------------------------------------------------------------------------
 if (bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation) //QBVH
 { //QBVH Rotation Order when Quaternions are invloved
     if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[3]==BVH_ROTATION_W) &&
          (BVH_ROTATION_ORDER_XYZ==enumerateRotationChannelOrderFromTypePositions(bvhMotion,currentJoint,4,5,6) )
        )
          {
              fprintf(stderr,GREEN "Root Quaternion detected..!\n" NORMAL);
              return BVH_ROTATION_ORDER_QWQXQYQZ;
          } else
     if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[0]==BVH_ROTATION_W) &&
          (BVH_ROTATION_ORDER_XYZ==enumerateRotationChannelOrderFromTypePositions(bvhMotion,currentJoint,1,2,3) )
        )
          {
              fprintf(stderr,GREEN "Joint Quaternion detected..!\n" NORMAL);
              return BVH_ROTATION_ORDER_QWQXQYQZ;
          }
 } else //QBVH Rotation Order when Quaternions are involved
 { // Regular rotation order resolution
   channelOrder=enumerateRotationChannelOrderFromTypePositions(bvhMotion,currentJoint,0,1,2);

   if (channelOrder==BVH_ROTATION_ORDER_NONE)
    {
      channelOrder=enumerateRotationChannelOrderFromTypePositions(bvhMotion,currentJoint,3,4,5);
      if (channelOrder==BVH_ROTATION_ORDER_NONE)
        {
            fprintf(stderr,"Failed to resolve rotation order.. :(\n");
        }
    }
 }  // Regular rotation order resolution

  if (channelOrder==BVH_ROTATION_ORDER_NONE)
  {
    fprintf(stderr,RED "BUG: Channel order still wrong, TODO smarter channel order enumeration..\n" NORMAL);
  }

 return channelOrder;
}
#else
int enumerateChannelOrder(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint)
{
  int channelOrder=BVH_ROTATION_ORDER_NONE;
  int quaternionUsed = 0;

  for (unsigned int i=0; i<bvhMotion->jointHierarchy[currentJoint].loadedChannels; i++)
  {
    if (bvhMotion->jointHierarchy[currentJoint].channelType[i]==BVH_ROTATION_W)
    {
      quaternionUsed=1;
    }
  }

  /*
  fprintf(stderr,GREEN "enumerateChannelOrder %s %s %s %s %s %s %s..!\n" NORMAL,
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[0]],
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[1]],
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[2]],
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[3]],
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[4]],
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[5]],
  channelNames[bvhMotion->jointHierarchy[currentJoint].channelType[6]]);
  */

 if (quaternionUsed) //QBVH
 { //QBVH Rotation Order when Quaternions are invloved
     if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[3]==BVH_ROTATION_W) &&
          (bvhMotion->jointHierarchy[currentJoint].channelType[4]==BVH_ROTATION_X) &&
          (bvhMotion->jointHierarchy[currentJoint].channelType[5]==BVH_ROTATION_Y) &&
          (bvhMotion->jointHierarchy[currentJoint].channelType[6]==BVH_ROTATION_Z)
        )
          {
              fprintf(stderr,GREEN "Root Quaternion detected ( %d/%d/%d/%d )..!\n" NORMAL,
                             bvhMotion->jointHierarchy[currentJoint].channelType[3],
                             bvhMotion->jointHierarchy[currentJoint].channelType[4],
                             bvhMotion->jointHierarchy[currentJoint].channelType[5],
                             bvhMotion->jointHierarchy[currentJoint].channelType[6]);

              bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=1; //The Rotation is on offsets 3-6 so there is also a positional channel
              bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
              bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=1;
              bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=0;
              return BVH_ROTATION_ORDER_QWQXQYQZ;
          } else
     if (
          (bvhMotion->jointHierarchy[currentJoint].channelType[0]==BVH_ROTATION_W) &&
          (bvhMotion->jointHierarchy[currentJoint].channelType[1]==BVH_ROTATION_X) &&
          (bvhMotion->jointHierarchy[currentJoint].channelType[2]==BVH_ROTATION_Y) &&
          (bvhMotion->jointHierarchy[currentJoint].channelType[3]==BVH_ROTATION_Z)
        )
          {
              fprintf(stderr,GREEN "Joint Quaternion detected..!\n" NORMAL);
              bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=0; //The Rotation is specified on offsets 0-3 so there is no positional channel
              bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
              bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=1;
              bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=0;
              return BVH_ROTATION_ORDER_QWQXQYQZ;
          }
 } else //QBVH Rotation Order when Quaternions are involved
 { // Regular rotation order resolution
   channelOrder=enumerateRotationChannelOrderFromTypes(
                                                       bvhMotion->jointHierarchy[currentJoint].channelType[0],
                                                       bvhMotion->jointHierarchy[currentJoint].channelType[1],
                                                       bvhMotion->jointHierarchy[currentJoint].channelType[2]
                                                      );

    if (channelOrder==BVH_ROTATION_ORDER_RODRIGUES)
    {
      bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=0; //The Rotation is on offsets 0-2 so there is no positional channel
      bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
      bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=0;
      bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=1;
    } else
    if (channelOrder!=BVH_ROTATION_ORDER_NONE)
    {
      bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=0; //The Rotation is on offsets 0-2 so there is no positional channel
      bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
      bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=0;
      bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=0;
    }
   else
   if (channelOrder==BVH_ROTATION_ORDER_NONE)
    {
      channelOrder=enumerateRotationChannelOrderFromTypes(
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[3],
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[4],
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[5]
                                                 );
    if (channelOrder==BVH_ROTATION_ORDER_RODRIGUES)
    {
      bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=1; //The Rotation is on offsets 0-2 so there is no positional channel
      bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
      bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=0;
      bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=1;
    } else
    if (channelOrder!=BVH_ROTATION_ORDER_NONE)
        {
          bvhMotion->jointHierarchy[currentJoint].hasPositionalChannels=1; //The Rotation is on offsets 0-2 so there is no positional channel(?)
          bvhMotion->jointHierarchy[currentJoint].hasRotationalChannels=1;
          bvhMotion->jointHierarchy[currentJoint].hasQuaternionRotation=0;
          bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation=0;
        } else
        {
            fprintf(stderr,"Failed to resolve rotation order.. :(\n");
        }
    }
 }  // Regular rotation order resolution

  if (channelOrder==BVH_ROTATION_ORDER_NONE)
  {
    fprintf(stderr,RED "BUG: Channel order still wrong, TODO smarter channel order enumeration..\n" NORMAL);
  }

 return channelOrder;
}
#endif // NEW_CHANNELROTATION_ENUMERATION_CODE




#define NEW_RESOLVE_CODE 1


#if NEW_RESOLVE_CODE
unsigned int bvh_resolveFrameAndJointAndChannelToMotionID(struct BVH_MotionCapture * bvhMotion,BVHJointID jID, BVHFrameID fID,unsigned int channelTypeID)
{
   unsigned int result = 0;
   //-----------------------
   if (
         (bvhMotion!=0) &&
         (channelTypeID<BVH_VALID_CHANNEL_NAMES) &&
         (jID<bvhMotion->jointHierarchySize) &&
         (fID<bvhMotion->numberOfFrames)
      )
   {
     //Manual re-resolution of channel!
     //-------------------------------------------------------------------------------------------------------------
     unsigned int resolvedChannel       = bvhMotion->jointHierarchy[jID].resolveChannelType[channelTypeID];
     unsigned int resolvedChannelType   = (resolvedChannel!=BVH_INVALID_CHANNEL);
     unsigned int channelIDMotionOffset = bvhMotion->jointToMotionLookup[jID].jointMotionOffset + resolvedChannel;
     //-------------------------------------------------------------------------------------------------------------
     result = (resolvedChannelType) * ((fID * bvhMotion->numberOfValuesPerFrame) + channelIDMotionOffset);
   }
  return result;
}
#else
unsigned int bvh_resolveFrameAndJointAndChannelToMotionID(struct BVH_MotionCapture * bvhMotion,BVHJointID jID, BVHFrameID fID,unsigned int channelTypeID)
{
   if (
         (bvhMotion!=0) &&
         (channelTypeID<BVH_VALID_CHANNEL_NAMES) &&
         (jID<bvhMotion->jointHierarchySize) &&
         (fID<bvhMotion->numberOfFrames) &&
         (bvhMotion->jointHierarchy[jID].loadedChannels < BVH_VALID_CHANNEL_NAMES)
      )
   {
     //Manual re-resolution of channel!
       int resolvedChannel     = 0;
       int resolvedChannelType = 0;
       for (int cID=0; cID<bvhMotion->jointHierarchy[jID].loadedChannels; cID++)
       {
        //fprintf(stderr,"LOOP jID %u / channel %u / channelTypeID %u\n",jID,cID,bvhMotion->jointHierarchy[jID].channelType[cID]);
        if (channelTypeID==bvhMotion->jointHierarchy[jID].channelType[cID])
        {
            resolvedChannelType = 1;
            resolvedChannel     = cID;
        }
       }

     if (resolvedChannelType!=0)
     {
      //unsigned int channelIDMotionOffset = bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[channelTypeID];
       unsigned int channelIDMotionOffset =  bvhMotion->jointToMotionLookup[jID].jointMotionOffset + resolvedChannel;
       return  (fID * bvhMotion->numberOfValuesPerFrame) + channelIDMotionOffset;
     }

     if ( (jID==0) && (channelTypeID==BVH_ROTATION_W) )
         {
           //Special case QBVH ..!
         } else
         {
          fprintf(
                  stderr,RED "BUG:%s|Unable to resolve jID %u (%s)|fID %u/%u|channelTypeID %u\n" NORMAL,
                  bvhMotion->fileName,
                  jID,
                  bvhMotion->jointHierarchy[jID].jointName,
                  fID,
                  bvhMotion->numberOfFrames,
                  channelTypeID
                 );
         }
     //This is broken ?
   }

  return 0;
}
#endif



//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        MOTION PARSING
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

int bvh_free(struct BVH_MotionCapture * bvhMotion)
{
  if (bvhMotion==0) { return 0; }
  //-----------------------------------------------------------------------------------------------------------------------
  if (bvhMotion->motionValues!=0)               { free(bvhMotion->motionValues);        bvhMotion->motionValues=0;        }
  if (bvhMotion->selectedJoints!=0)             { free(bvhMotion->selectedJoints);      bvhMotion->selectedJoints=0;      }
  if (bvhMotion->hideSelectedJoints!=0)         { free(bvhMotion->hideSelectedJoints);  bvhMotion->hideSelectedJoints=0;  }
  if (bvhMotion->fileName!=0)                   { free(bvhMotion->fileName);            bvhMotion->fileName=0;            }
  //-----------------------------------------------------------------------------------------------------------------------
  if (bvhMotion->jointHierarchy!=0)             { free(bvhMotion->jointHierarchy);      bvhMotion->jointHierarchy=0;      }
  if (bvhMotion->jointToMotionLookup!=0)        { free(bvhMotion->jointToMotionLookup); bvhMotion->jointToMotionLookup=0; }
  if (bvhMotion->motionToJointLookup!=0)        { free(bvhMotion->motionToJointLookup); bvhMotion->motionToJointLookup=0; }
  //-----------------------------------------------------------------------------------------------------------------------

  //Wipe everything
  memset(bvhMotion,0,sizeof(struct BVH_MotionCapture));

  return 1;
}

//----------------------------------------------------------------------------------------------------
int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion, float scaleWorld)
{
  //fprintf(stderr,"Loading BVH %s \n",filename);
  bvhMotion->scaleWorld=scaleWorld;
  int successfullRead=0;
  FILE *fd=0;
  fd = fopen(filename,"r");
  if (fd!=0)
    {
      //Remember filename..!
      //==========================================================================
      unsigned int lengthOfFilename = strlen(filename)+1; //+1 to be null terminated..
      if (bvhMotion->fileName!=0) { free(bvhMotion->fileName); }
      bvhMotion->fileName = (char *) malloc(sizeof(char) * (lengthOfFilename));
      if (bvhMotion->fileName!=0)
        { snprintf(bvhMotion->fileName,lengthOfFilename,"%s",filename); }
      //==========================================================================

      if (readBVHHeader(bvhMotion,fd))
      {
       //If we have the header let's update the hashes
       bvh_updateJointNameHashes(bvhMotion);

       if (readBVHMotion(bvhMotion,fd))
       {
         successfullRead=1;
       } else
       { fprintf(stderr,"Error reading BVH motion for file %s\n",filename); }
      } else
      { fprintf(stderr,"Error reading BVH header for file %s\n",filename); }
      fclose(fd);
    } else
    { fprintf(stderr,"Could not open BVH file %s to read\n",filename); }

 return successfullRead;
}
//----------------------------------------------------------------------------------------------------


int bvh_setMIDValue(
                    struct BVH_MotionCapture * mc,
                    unsigned int mID,
                    float value
                   )
{
  if (mc!=0)
  {
   unsigned int fID=0;
   for (fID=0; fID<mc->numberOfFrames; fID++)
    {
     unsigned int absMID=mID+(fID*mc->numberOfValuesPerFrame);
     mc->motionValues[absMID]=value;
    }
   return 1;
  }
 return 0;
}




int bvh_SetPositionRotation(
                             struct BVH_MotionCapture * mc,
                             struct motionTransactionData * positionAndRotation
                           )
{
  //---------------------------------------------------------------------------------------------------------------
  unsigned int positionXOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_POSITION_X);
  unsigned int positionYOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_POSITION_Y);
  unsigned int positionZOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_POSITION_Z);
  //---------------------------------------------------------------------------------------------------------------
  unsigned int rotationXOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_X);
  unsigned int rotationYOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_Y);
  unsigned int rotationZOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_Z);
  //---------------------------------------------------------------------------------------------------------------

  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   mc->motionValues[mID+positionXOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X];
   mc->motionValues[mID+positionYOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y];
   mc->motionValues[mID+positionZOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z];
   mc->motionValues[mID+rotationXOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X];
   mc->motionValues[mID+rotationYOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y];
   mc->motionValues[mID+rotationZOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z];
  }

  //Special case when we have a root quaternion, we need to also update it..!
  if (mc->jointHierarchy[mc->rootJointID].hasQuaternionRotation)
  {
    unsigned int rotationWOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_W);
    for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mID=fID*mc->numberOfValuesPerFrame;
       mc->motionValues[mID+rotationWOffset]=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W];
      }
  }
 return 1;
}


int bvh_OffsetPositionRotation(
                               struct BVH_MotionCapture * mc,
                               struct motionTransactionData * positionAndRotation
                              )
{
  //---------------------------------------------------------------------------------------------------------------
  unsigned int positionXOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_POSITION_X);
  unsigned int positionYOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_POSITION_Y);
  unsigned int positionZOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_POSITION_Z);
  //---------------------------------------------------------------------------------------------------------------
  unsigned int rotationXOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_X);
  unsigned int rotationYOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_Y);
  unsigned int rotationZOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_Z);
  //---------------------------------------------------------------------------------------------------------------

  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   mc->motionValues[mID+positionXOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X];
   mc->motionValues[mID+positionYOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y];
   mc->motionValues[mID+positionZOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z];
   mc->motionValues[mID+rotationXOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X];
   mc->motionValues[mID+rotationYOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y];
   mc->motionValues[mID+rotationZOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z];
  }


  //Special case when we have a root quaternion, we need to also update it..!
  if (mc->jointHierarchy[mc->rootJointID].hasQuaternionRotation)
  {
    unsigned int rotationWOffset = bvh_resolveFrameAndJointAndChannelToMotionID(mc,mc->rootJointID,0,BVH_ROTATION_W);
    for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mID=fID*mc->numberOfValuesPerFrame;
       mc->motionValues[mID+rotationWOffset]+=positionAndRotation->data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W];
      }
  }
 return 1;
}




//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                           ACCESSORS
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
int bvh_jointHasParent(struct BVH_MotionCapture * bvhMotion,BVHJointID jID)
{
 if (jID<bvhMotion->jointHierarchySize)
     {
       return (!bvhMotion->jointHierarchy[jID].isRoot);
     }
 return 0;
}

int bvh_jointGetEndSiteChild(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,BVHJointID * jChildID)
{
  //fprintf(stderr," bvh_jointGetEndSiteChild ");
  if (bvhMotion==0) { return 0; }

  if (bvhMotion->jointHierarchy[jID].hasEndSite)
  {
   unsigned int jointID=0;
   for (jointID=0; jointID<bvhMotion->jointHierarchySize; jointID++)
   {
     //fprintf(stderr,"joint[%u]=%s ",jointID,bvhMotion->jointHierarchy[jointID].jointName);
     if (bvhMotion->jointHierarchy[jointID].isEndSite)
     {
          if (bvhMotion->jointHierarchy[jointID].parentJoint==jID)
          {
              *jChildID=jointID;
              return 1;
          }
     }
   }
  }
 return 0;
}


int bvh_jointHasRotation(struct BVH_MotionCapture * bvhMotion , BVHJointID jID)
{
 if (bvhMotion)
 {
  if (jID<bvhMotion->jointHierarchySize)
     {
        return bvhMotion->jointHierarchy[jID].hasRotationalChannels;
     }
 }
 return 0;
}



int bvh_getJointIDFromJointName(
                                 struct BVH_MotionCapture * bvhMotion ,
                                 const char * jointName,
                                 BVHJointID * jID
                                )
{
   if ( (bvhMotion!=0) && (jointName!=0) && (jID!=0) )
   {
    unsigned int i=0;
    for (i=0; i<bvhMotion->jointHierarchySize; i++)
    {
     if (strcmp(bvhMotion->jointHierarchy[i].jointName,jointName)==0)
     {
         *jID=i;
         return 1;
     }
    }
   }
 return 0;
}



int bvh_getJointIDFromJointNameNocase(
                                      struct BVH_MotionCapture * bvhMotion ,
                                      const char * jointName,
                                      BVHJointID * jID
                                     )
{
 if ( (bvhMotion!=0) && (jointName!=0) && (jID!=0) && (bvhMotion->jointHierarchy!=0) )
 {
   unsigned int jointNameLength = strlen(jointName);
   //-----------------------------------------------
   if (jointNameLength==0)
     {
       fprintf(stderr,"bvh_getJointIDFromJointNameNocase failed because of 0 joint name..");
     } else
   if (jointNameLength<MAX_BVH_JOINT_NAME)
     {
       //Moved to heap @ 2021/04/21 trying to debug a stack overflow.. :P
       //char jointNameLowercase[MAX_BVH_JOINT_NAME+1]={0};
       char * jointNameLowercase = (char *) malloc(sizeof(char) * (MAX_BVH_JOINT_NAME+1)); //extra space for the null termination..

       if (jointNameLowercase!=0)
       {
        snprintf(jointNameLowercase,MAX_BVH_JOINT_NAME,"%s",jointName);
        lowercase(jointNameLowercase);

        unsigned int i=0;
        for (i=0; i<bvhMotion->jointHierarchySize; i++)
         {
           if (strcmp(bvhMotion->jointHierarchy[i].jointNameLowercase,jointNameLowercase)==0)
           {
            *jID=i;
            free(jointNameLowercase);
            return 1;
           }
          }
       free(jointNameLowercase);
      }
     } else
     {
       fprintf(stderr,"bvh_getJointIDFromJointNameNocase failed because of very long joint names..");
     }
 }
 return 0;
}


int bvh_getRootJointID(
                       struct BVH_MotionCapture * bvhMotion,
                       BVHJointID * jID
                      )
{
  int res = 0;
  if ( (bvhMotion!=0) && (jID!=0) )
  {
   *jID = bvhMotion->rootJointID;
   res = 1;
  }

  return res;
}


int bvh_getJointParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID)
{
  int res = 0;
  if ( (bvhMotion!=0) && (jID<bvhMotion->jointHierarchySize) )
  {
       res = bvhMotion->jointHierarchy[jID].parentJoint;
  }
 return res;
}





int bvh_isJointAChildrenID(
                           struct BVH_MotionCapture * bvhMotion,
                           BVHJointID parentJID,
                           BVHJointID childJID
                          )
{
  int res = 0;
  if ( (bvhMotion!=0) && (parentJID!=0)  && (childJID!=0) )
  {
   unsigned int jumps = 0;

   BVHJointID jID = childJID;
    while (jID!=bvhMotion->rootJointID)
    {
      if(bvhMotion->jointHierarchy[jID].parentJoint == parentJID) { res=1; break; }
      if(jID == parentJID)                                        { res=1; break; }

      //Jump to parent..
      jID = bvhMotion->jointHierarchy[jID].parentJoint;

       if (jumps > bvhMotion->jointHierarchySize)
                  {
                    fprintf(stderr,RED "BUG: more jumps than hierarchy size ?" NORMAL);
                    res=0;
                    break;
                  }
      ++jumps;
    }
  }
  return res;
}


int bvh_onlyAnimateGivenJoints(struct BVH_MotionCapture * bvhMotion,unsigned int numberOfArguments,const char **argv)
{
    bvh_printBVH(bvhMotion);
    fprintf(stderr,"bvh_onlyAnimateGivenJoints with %u arguments\n",numberOfArguments);

    BVHJointID * activeJoints = (BVHJointID*) malloc(sizeof(BVHJointID) * numberOfArguments);
    //-----------------------------------------------------------------------------------------------
    if (activeJoints==0)
    {
      fprintf(stderr,"bvh_onlyAnimateGivenJoints failed to allocate space for %u arguments\n",numberOfArguments);
      return 0;
    } else
    {
      memset(activeJoints,0,sizeof(BVHJointID) * numberOfArguments);
    }



    char * successJoints = (char *) malloc(sizeof(char) * numberOfArguments);
    //-----------------------------------------------------------------------------------------------
    if (successJoints==0)
    {
      fprintf(stderr,"bvh_onlyAnimateGivenJoints failed to allocate space for %u arguments\n",numberOfArguments);
      free(activeJoints);
      return 0;
    } else
    {
     memset(successJoints,0,sizeof(char) * numberOfArguments);
    }

    if ((activeJoints!=0) && (successJoints!=0))
    {

    for (unsigned int i=0; i<numberOfArguments; i++)
    {
      BVHJointID jID=0;

      if (
           bvh_getJointIDFromJointNameNocase(
                                             bvhMotion ,
                                             argv[i],
                                             &jID
                                            )
         )
         {
           fprintf(stderr,GREEN "Joint Activated %u = %s -> jID=%u\n" NORMAL,i,argv[i],jID);
           activeJoints[i]=jID;
           successJoints[i]=1;
         } else
         {
           fprintf(stderr,RED "Joint Failed to Activate %u = %s\n" NORMAL,i,argv[i]);
           fprintf(stderr,RED "Check the list above to find correct joint names..\n" NORMAL);
         }
    }


      unsigned int mID_Initial,mID_Target;
      for (unsigned int frameID=0; frameID<bvhMotion->numberOfFramesEncountered; frameID++)
       {
         //fprintf(stderr,"FrameNumber %u\n",frameID);
         for (unsigned int mID=0; mID<bvhMotion->numberOfValuesPerFrame; mID++)
         {
             unsigned int  jointID = bvhMotion->motionToJointLookup[mID].jointID;
             int isMIDProtected=0;

             for (unsigned int aJ=0; aJ<numberOfArguments; aJ++)
              {
               if (successJoints[aJ])
                {
                   if (jointID==activeJoints[aJ])
                   {
                     isMIDProtected=1;
                   }
                }
              }

            if (!isMIDProtected)
            {
              mID_Initial=mID;
              mID_Target=frameID * bvhMotion->numberOfValuesPerFrame + mID;
              bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];
            }

         }

         /* This does the inverse
         unsigned int firstFrame=0;
         for (int aJ=0; aJ<numberOfArguments; aJ++)
         {
           if (successJoints[aJ])
           {
            jointID = activeJoints[aJ];

            mID_Initial = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,firstFrame,BVH_ROTATION_X);
            mID_Target = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,frameID,BVH_ROTATION_X);
            bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];

            mID_Initial = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,firstFrame,BVH_ROTATION_Y);
            mID_Target = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,frameID,BVH_ROTATION_Y);
            bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];

            mID_Initial = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,firstFrame,BVH_ROTATION_Z);
            mID_Target = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,frameID,BVH_ROTATION_Z);
            bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];
           }
         }
         */
       }
      free(successJoints);
      free(activeJoints);
      return 1;
    }

  return 0;
}







int bvh_getJointDimensions(
                              struct BVH_MotionCapture * bvhMotion,
                              const char * jointName,
                              float * xValue,
                              float * yValue,
                              float * zValue
                             )
{
   if (bvhMotion!=0)
    {
      BVHJointID jID=0;
      if ( bvh_getJointIDFromJointNameNocase(bvhMotion,jointName,&jID) )
      //if ( bvh_getJointIDFromJointName(bvhMotion ,jointName,&jID) )
      {
       unsigned int angleX = 0;
       unsigned int angleY = 1;
       unsigned int angleZ = 2;

       for (int ch=0; ch<3; ch++)
       {
           //TODO: add qbvh support
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_X) { angleX = ch; }
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_Y) { angleY = ch; }
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_Z) { angleZ = ch; }
       }

       if (xValue!=0) { *xValue = bvhMotion->jointHierarchy[jID].offset[angleX]; }
       if (yValue!=0) { *yValue = bvhMotion->jointHierarchy[jID].offset[angleY]; }
       if (zValue!=0) { *zValue = bvhMotion->jointHierarchy[jID].offset[angleZ]; }

      return 1;
      }
   }
 return 0;
}





int bvh_changeJointDimensions(
                              struct BVH_MotionCapture * bvhMotion,
                              const char * jointName,
                              float xScale,
                              float yScale,
                              float zScale
                             )
{
 if (bvhMotion!=0)
    {
      //fprintf(stderr,"bvh_changeJointDimensions: %s %0.2f %0.2f %0.2f\n",jointName,xScale,yScale,zScale);

      BVHJointID jID=0;
      if ( bvh_getJointIDFromJointNameNocase(bvhMotion,jointName,&jID) )
      //if ( bvh_getJointIDFromJointName(bvhMotion ,jointName,&jID) )
      {
       unsigned int angleX = 0;
       unsigned int angleY = 1;
       unsigned int angleZ = 2;

       for (int ch=0; ch<3; ch++)
       {
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_X) { angleX = ch; }
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_Y) { angleY = ch; }
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_Z) { angleZ = ch; }
       }

       //fprintf(stderr,"offset was %0.2f %0.2f %0.2f\n",bvhMotion->jointHierarchy[jID].offset[0],bvhMotion->jointHierarchy[jID].offset[1],bvhMotion->jointHierarchy[jID].offset[2]);
       bvhMotion->jointHierarchy[jID].offset[angleX] *= xScale;
       bvhMotion->jointHierarchy[jID].offset[angleY] *= yScale;
       bvhMotion->jointHierarchy[jID].offset[angleZ] *= zScale;
       //fprintf(stderr,"offset is now %0.2f %0.2f %0.2f\n",bvhMotion->jointHierarchy[jID].offset[0],bvhMotion->jointHierarchy[jID].offset[1],bvhMotion->jointHierarchy[jID].offset[2]);

       float * m = bvhMotion->jointHierarchy[jID].staticTransformation.m;
       m[0] =1.0;  m[1] =0.0;  m[2] =0.0;  m[3] = (float) bvhMotion->jointHierarchy[jID].offset[0];
       m[4] =0.0;  m[5] =1.0;  m[6] =0.0;  m[7] = (float) bvhMotion->jointHierarchy[jID].offset[1];
       m[8] =0.0;  m[9] =0.0;  m[10]=1.0;  m[11]= (float) bvhMotion->jointHierarchy[jID].offset[2];
       m[12]=0.0;  m[13]=0.0;  m[14]=0.0;  m[15]=1.0;

      return 1;
    }

    fprintf(stderr,"bvh_changeJointDimensions: Unable to locate joint %s \n",jointName);
    fprintf(stderr,"Joint List : ");
    for (unsigned int jIDIt=0; jIDIt<bvhMotion->jointHierarchySize; jIDIt++)
     {
       if (jIDIt!=0) { fprintf(stderr,","); }
       fprintf(stderr,"%s(or. %s)",bvhMotion->jointHierarchy[jIDIt].jointNameLowercase,bvhMotion->jointHierarchy[jIDIt].jointName);
     }
    fprintf(stderr,"\n");
   }
 return 0;
}



int bvh_scaleAllOffsets(
                        struct BVH_MotionCapture * bvhMotion,
                        float scalingRatio
                       )
{
  if (scalingRatio==1.0) { return 1; }
  if (bvhMotion!=0)
    {
     for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
     {
      bvhMotion->jointHierarchy[jID].offset[0] = bvhMotion->jointHierarchy[jID].offset[0] * scalingRatio;
      bvhMotion->jointHierarchy[jID].offset[1] = bvhMotion->jointHierarchy[jID].offset[1] * scalingRatio;
      bvhMotion->jointHierarchy[jID].offset[2] = bvhMotion->jointHierarchy[jID].offset[2] * scalingRatio;

       float * m = bvhMotion->jointHierarchy[jID].staticTransformation.m;
       m[0] =1.0;  m[1] =0.0;  m[2] =0.0;  m[3] = (float) bvhMotion->jointHierarchy[jID].offset[0];
       m[4] =0.0;  m[5] =1.0;  m[6] =0.0;  m[7] = (float) bvhMotion->jointHierarchy[jID].offset[1];
       m[8] =0.0;  m[9] =0.0;  m[10]=1.0;  m[11]= (float) bvhMotion->jointHierarchy[jID].offset[2];
       m[12]=0.0;  m[13]=0.0;  m[14]=0.0;  m[15]=1.0;
     }
     return 1;
    }
  return 0;
}


int bvh_scaleAllJointChildrenOffsets(
                                     struct BVH_MotionCapture * bvhMotion,
                                     const char * jointName,
                                     float scalingRatio
                                    )
{
  if (scalingRatio==1.0) { return 1; }
  if (bvhMotion!=0)
    {
      BVHJointID parentID=0;
      if ( bvh_getJointIDFromJointNameNocase(bvhMotion,jointName,&parentID) )
      {
        for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
        {
         if ( bvh_isJointAChildrenID(bvhMotion,parentID,jID) )
         {
          bvhMotion->jointHierarchy[jID].offset[0] = bvhMotion->jointHierarchy[jID].offset[0] * scalingRatio;
          bvhMotion->jointHierarchy[jID].offset[1] = bvhMotion->jointHierarchy[jID].offset[1] * scalingRatio;
          bvhMotion->jointHierarchy[jID].offset[2] = bvhMotion->jointHierarchy[jID].offset[2] * scalingRatio;

          float * m = bvhMotion->jointHierarchy[jID].staticTransformation.m;
          m[0] =1.0;  m[1] =0.0;  m[2] =0.0;  m[3] = (float) bvhMotion->jointHierarchy[jID].offset[0];
          m[4] =0.0;  m[5] =1.0;  m[6] =0.0;  m[7] = (float) bvhMotion->jointHierarchy[jID].offset[1];
          m[8] =0.0;  m[9] =0.0;  m[10]=1.0;  m[11]= (float) bvhMotion->jointHierarchy[jID].offset[2];
          m[12]=0.0;  m[13]=0.0;  m[14]=0.0;  m[15]=1.0;
         }
       }
       return 1;
      }
    }
  return 0;
}



int bvh_getMotionChannelName(struct BVH_MotionCapture * bvhMotion,BVHMotionChannelID mID,char * target,unsigned int targetLength)
{
 if (mID<bvhMotion->numberOfValuesPerFrame)
 {
   BVHJointID jID = bvhMotion->motionToJointLookup[mID].jointID;
   unsigned int channelID = bvhMotion->motionToJointLookup[mID].channelID;
   if ( (jID<bvhMotion->jointHierarchySize) && (channelID<BVH_VALID_CHANNEL_NAMES) )
   {
      char * jointLabel = bvhMotion->jointHierarchy[jID].jointName;

      int i = snprintf(target,targetLength,"%s_%s",jointLabel,channelNames[channelID]);
      if (i<0)
      {
         fprintf(stderr,"Not enough space to hold motion channel name for mID=%u\n",mID);
         return 0;
      }
   }
 }
 return 0;
}








//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
float bvh_getJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID)
{
   if ( (bvhMotion!=0) && (jID<bvhMotion->jointHierarchySize) )
    {
      unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

      if (mID<bvhMotion->motionValuesSize)
       {
         return bvhMotion->motionValues[mID];
       } else
       {
         //bvh_printBVH(bvhMotion);
         fprintf(stderr,RED "bvh_getJointChannelAtFrame overflowed..\n" NORMAL);
       }
    }
 return 0.0;
}

float  bvh_getJointRotationWAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_W); }
float  bvh_getJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_X); }
float  bvh_getJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Y); }
float  bvh_getJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Z); }
float  bvh_getJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_X); }
float  bvh_getJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Y); }
float  bvh_getJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Z); }

int bvh_populatePosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData)
{
 if ( (data!=0) && (sizeOfData >= sizeof(float) * MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER) ) //QBVH
  {
  data[BVH_POSITION_X]=bvh_getJointPositionXAtFrame(bvhMotion,jID,fID);
  data[BVH_POSITION_Y]=bvh_getJointPositionYAtFrame(bvhMotion,jID,fID);
  data[BVH_POSITION_Z]=bvh_getJointPositionZAtFrame(bvhMotion,jID,fID);
  data[BVH_ROTATION_W]=bvh_getJointRotationWAtFrame(bvhMotion,jID,fID);
  data[BVH_ROTATION_X]=bvh_getJointRotationXAtFrame(bvhMotion,jID,fID);
  data[BVH_ROTATION_Y]=bvh_getJointRotationYAtFrame(bvhMotion,jID,fID);
  data[BVH_ROTATION_Z]=bvh_getJointRotationZAtFrame(bvhMotion,jID,fID);
  return 1;
  }
  return 0;
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------



//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
int bvh_setJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID,float value)
{
   int result = 0;
   if ( (bvhMotion!=0) && (jID<bvhMotion->jointHierarchySize) )
   {
     unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);
     if (mID<bvhMotion->motionValuesSize)
     {
       bvhMotion->motionValues[mID]=value;
       result=1;
     } else
     { fprintf(stderr,RED "bvh_setJointChannelAtFrame overflowed..\n" NORMAL); }
   }
   return result;
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
int bvh_setJointRotationWAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_W,value); }
int bvh_setJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_X,value); }
int bvh_setJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Y,value); }
int bvh_setJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Z,value); }
int bvh_setJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_X,value); }
int bvh_setJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Y,value); }
int bvh_setJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Z,value); }
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------

int bvh_setPosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData)
{
  if ( (bvhMotion!=0) && (data!=0) && (sizeOfData >= sizeof(float) * MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER) ) //QBVH
  {
   int successfulStores=bvh_setJointPositionXAtFrame(bvhMotion,jID,fID,data[BVH_POSITION_X]);
   successfulStores+=bvh_setJointPositionYAtFrame(bvhMotion,jID,fID,data[BVH_POSITION_Y]);
   successfulStores+=bvh_setJointPositionZAtFrame(bvhMotion,jID,fID,data[BVH_POSITION_Z]);
   successfulStores+=bvh_setJointRotationXAtFrame(bvhMotion,jID,fID,data[BVH_ROTATION_W]);
   successfulStores+=bvh_setJointRotationXAtFrame(bvhMotion,jID,fID,data[BVH_ROTATION_X]);
   successfulStores+=bvh_setJointRotationYAtFrame(bvhMotion,jID,fID,data[BVH_ROTATION_Y]);
   successfulStores+=bvh_setJointRotationZAtFrame(bvhMotion,jID,fID,data[BVH_ROTATION_Z]);
   return (successfulStores==7);
  }
  return 0;
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
float bvh_getJointChannelAtMotionBuffer(struct BVH_MotionCapture * bvhMotion, BVHJointID jID,float * motionBuffer, unsigned int channelTypeID)
{
   if (bvhMotion!=0)
   {
       if (jID<bvhMotion->jointHierarchySize)
       {
         unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,channelTypeID);

         if (mID<bvhMotion->motionValuesSize)
           {
             return motionBuffer[mID];
           }
         fprintf(stderr,RED "bvh_getJointChannelAtMotionBuffer error ( tried to access mID %u/%u )..\n" NORMAL,mID,bvhMotion->motionValuesSize);
       } else
       {
         fprintf(stderr,RED "bvh_getJointChannelAtMotionBuffer error ( tried to access jID %u/%u )..\n" NORMAL,jID,bvhMotion->MAX_jointHierarchySize);
       }
   }
 return 0.0;
}

float  bvh_getJointRotationWAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_W); } //QBVH
float  bvh_getJointRotationXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_X); }
float  bvh_getJointRotationYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_Y); }
float  bvh_getJointRotationZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_Z); }
float  bvh_getJointPositionXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_X); }
float  bvh_getJointPositionYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_Y); }
float  bvh_getJointPositionZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_Z); }


int bvh_retrieveDataFromMotionBuffer(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , float * motionBuffer, float * data, unsigned int sizeOfData)
{
  int res = 0;
  //This gets spammed a *LOT* so it needs to be improved..
  if ( (motionBuffer!=0) && (data!=0) && (sizeOfData >= sizeof(float) * MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER) ) //QBVH
  {
      //?
      //if (bvhMotion->jointHierarchy[jID].isEndSite)
      //{
        //Fast-Path out, end-sites have no data to retrieve
      //  return 1;
      //}

      // If there are no positional channels erase them..!
      if (!bvhMotion->jointHierarchy[jID].hasPositionalChannels) //This used to be isRoot before QBVH
      {
       data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X]=0.0;
       data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y]=0.0;
       data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z]=0.0;
      } else
      {
       //Only Root joint has a position field..
       //fprintf(stderr,"jID %u (%s) has a position field..\n",jID,bvhMotion->jointHierarchy[jID].jointName);
       data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X]=bvh_getJointPositionXAtMotionBuffer(bvhMotion,jID,motionBuffer);
       data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y]=bvh_getJointPositionYAtMotionBuffer(bvhMotion,jID,motionBuffer);
       data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z]=bvh_getJointPositionZAtMotionBuffer(bvhMotion,jID,motionBuffer);
      }

    if (bvhMotion->jointHierarchy[jID].hasRotationalChannels) //This used to be isRoot before QBVH
      {
       #ifdef NAN
          // NAN is supported
          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W]=NAN;
       #else
          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W]=0.0;
       #endif

       unsigned int mID;

       if (bvhMotion->jointHierarchy[jID].channelRotationOrder==BVH_ROTATION_ORDER_ZXY)
       { //FAST PATH, this gets called millions of times..!
             //Special code to speed up cases that match the BVH specification ZXY rotation orders
             mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,BVH_ROTATION_X);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]=motionBuffer[mID];
             mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,BVH_ROTATION_Y);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]=motionBuffer[mID];
             mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,BVH_ROTATION_Z);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]=motionBuffer[mID];
       } else
       {//Slow path
         switch (bvhMotion->jointHierarchy[jID].channelRotationOrder)
         {
           case BVH_ROTATION_ORDER_NONE:
             fprintf(stderr,"bvh_retrieveDataFromMotionBuffer: Asked to retrieve without rotation order..!\n");
           break;

           case BVH_ROTATION_ORDER_QWQXQYQZ : //QBVH
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W]=bvh_getJointRotationWAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]=bvh_getJointRotationXAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]=bvh_getJointRotationYAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]=bvh_getJointRotationZAtMotionBuffer(bvhMotion,jID,motionBuffer);
           break;

           default :
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]=bvh_getJointRotationXAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]=bvh_getJointRotationYAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]=bvh_getJointRotationZAtMotionBuffer(bvhMotion,jID,motionBuffer);
           break;
         };
       } //End of slow path
      } else
      {
         //This is the case where a joint has no rotational channels!
         #ifdef NAN
          // NAN is supported
          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W]=NAN;
         #else
          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W]=0.0;
         #endif

          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]=0.0;
          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]=0.0;
          data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]=0.0;
      }

    res = 1;
  }

  return res;
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------

float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion,BVHMotionChannelID mID)
{
  if (bvhMotion!=0)
  {
   if (mID<bvhMotion->motionValuesSize)
   {
     return bvhMotion->motionValues[mID];
   }
  }
  fprintf(stderr,"Could not resolve mID %u\n",mID);
  return 0.0;
}


int bvh_setMotionValue(struct BVH_MotionCapture * bvhMotion,BVHMotionChannelID mID,float * value)
{
  if (bvhMotion!=0)
  {
   if (mID<bvhMotion->motionValuesSize)
   {
    bvhMotion->motionValues[mID]=*value;
    return 1;
   }
  }
  fprintf(stderr,"Could not resolve mID %u\n",mID);
  return 0;
}


float bvh_getMotionValueOfFrame(struct BVH_MotionCapture * bvhMotion,BVHFrameID fID,BVHMotionChannelID mID)
{
  if (bvhMotion!=0)
  {
   BVHMotionChannelID resolvedMID = fID*bvhMotion->numberOfValuesPerFrame + mID;
   return  bvh_getMotionValue(bvhMotion,resolvedMID);
  }
  return 0.0;
}

int bvh_setMotionValueOfFrame(struct BVH_MotionCapture * bvhMotion,BVHFrameID fID,BVHMotionChannelID mID,float * value)
{
  if (bvhMotion!=0)
  {
   BVHMotionChannelID resolvedMID = fID*bvhMotion->numberOfValuesPerFrame + mID;
   return bvh_setMotionValue(bvhMotion,resolvedMID,value);
  }
  return 0;
}

int bvh_copyMotionFrameToMotionBuffer(
                                       struct BVH_MotionCapture * bvhMotion,
                                       struct MotionBuffer * motionBuffer,
                                       BVHFrameID fromfID
                                     )
{
 if ( (motionBuffer!=0) && (motionBuffer->motion!=0) && ( bvhMotion->numberOfValuesPerFrame <= motionBuffer->bufferSize) && (fromfID < bvhMotion->numberOfFrames) )
   {
     memcpy(
             motionBuffer->motion,
             &bvhMotion->motionValues[fromfID * bvhMotion->numberOfValuesPerFrame],
             bvhMotion->numberOfValuesPerFrame * sizeof(float)
           );
     return 1;
   }
 return 0;
}




int bvh_copyMotionBufferToMotionFrame(
                                       struct BVH_MotionCapture * bvhMotion,
                                       BVHFrameID fromfID,
                                       struct MotionBuffer * motionBuffer
                                     )
{
 if ( (motionBuffer!=0) && (motionBuffer->motion!=0) && ( bvhMotion->numberOfValuesPerFrame <= motionBuffer->bufferSize) && (fromfID < bvhMotion->numberOfFrames) )
   {
     memcpy(
             &bvhMotion->motionValues[fromfID * bvhMotion->numberOfValuesPerFrame],
             motionBuffer->motion,
             bvhMotion->numberOfValuesPerFrame * sizeof(float)
           );
     return 1;
   }
 return 0;
}







int bvh_copyMotionFrame(
                         struct BVH_MotionCapture * bvhMotion,
                         BVHFrameID tofID,
                         BVHFrameID fromfID
                        )
{
   if (
         (tofID<bvhMotion->numberOfFrames ) && (fromfID<bvhMotion->numberOfFrames )
      )
   {
     memcpy(
             &bvhMotion->motionValues[tofID * bvhMotion->numberOfValuesPerFrame],
             &bvhMotion->motionValues[fromfID * bvhMotion->numberOfValuesPerFrame],
             bvhMotion->numberOfValuesPerFrame * sizeof(float)
           );
     return 1;
   }
 return 0;
}







int bvh_selectChildrenOfJoint(struct BVH_MotionCapture * mc, const char * parentJoint)
{
   if (mc==0)                 { return 0; }
   if (mc->jointHierarchy==0) { return 0; }
   if (parentJoint==0)        { return 0; }
   //=======================================

   BVHJointID parentJID=0;

   if (
       !bvh_getJointIDFromJointNameNocase(
                                           mc,
                                           parentJoint,
                                           &parentJID
                                         )
      )
      {
        fprintf(stderr,RED "bvh_selectChildrenOfJoint: Could not resolve joint %s..\n" NORMAL,parentJoint);
        return 0;
      }



     if (mc->selectedJoints!=0) {
                                   fprintf(stderr,YELLOW "Multiple selection of joints taking place..\n" NORMAL);
                                } else
                                {
                                 mc->selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
                                 if (mc->selectedJoints==0)
                                   {
                                     fprintf(stderr,RED "bvh_selectChildrenOfJoint failed to allocate selectedJoints\n" NORMAL);
                                     return 0;
                                   }
                                 }

   //This select erases the previous one..
   memset(mc->selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
   for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
   {
       mc->selectedJoints[jID] = bvh_isJointAChildrenID(mc,parentJID,jID);
   }

  return 1;
}





void bvh_considerIfJointIsSelected(
                                   struct BVH_MotionCapture * mc,
                                   unsigned int jID,
                                   int * isJointSelected,
                                   int * isJointEndSiteSelected
                                  )
{
    *isJointSelected=1;
    *isJointEndSiteSelected=1;

    //First of all, if no joint selections have occured then everything is selected..
    if (mc->selectedJoints)
    {
     //If we reached this far it means there is a selection active..
     //We consider everything unselected unless proven otherwise..
     *isJointSelected=0;
     *isJointEndSiteSelected=0;

     //We now check if this joint is selected..
     //-------------------------------------------------------------
     //If there is a selection declared then let's consider if the joint is selected..
     if (mc->jointHierarchy[jID].isEndSite)
     {
       //If we are talking about an endsite we will have to check with it's parent joint..
       unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
       if ( (mc->selectedJoints[parentID]) && (mc->selectionIncludesEndSites) )
                          { *isJointEndSiteSelected=1; }
     } else
     {
       //This is a regular joint..
       if (mc->selectedJoints[jID])
                          { *isJointSelected=1; }
     }
    }
}





int bvh_selectJoints(
                     struct BVH_MotionCapture * mc,
                     unsigned int numberOfValues,
                     unsigned int includeEndSites,
                     const char **argv,
                     unsigned int iplus1
                    )
{
  fprintf(stderr,"Asked to select %u Joints\n",numberOfValues);

  mc->selectionIncludesEndSites=includeEndSites;
  mc->numberOfJointsWeWantToSelect=numberOfValues;

  //Uncomment to force each call to selectJoints to invalidade previous calls..
  //if (mc->selectedJoints!=0) { free(mc->selectedJoints); mc->selectedJoints=0; }
  if (mc->selectedJoints!=0) {
                               fprintf(stderr,YELLOW "Multiple selection of joints taking place..\n" NORMAL);
                             } else
                             {
                              mc->selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
                              if (mc->selectedJoints==0)
                                   {
                                     fprintf(stderr,RED "bvh_selectJoints failed to allocate selectedJoints\n" NORMAL);
                                     return 0;
                                   } else
                                   {
                                      memset(mc->selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
                                   }
                             }


  //if (mc->hideSelectedJoints!=0) { free(mc->hideSelectedJoints); mc->hideSelectedJoints=0; }
   if (mc->hideSelectedJoints!=0) {
                                    fprintf(stderr,YELLOW "Multiple selection of hidden joints taking place..\n" NORMAL);
                                  } else
                                  {
                                     mc->hideSelectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
                                     if (mc->hideSelectedJoints==0)
                                          {
                                            fprintf(stderr,RED "bvh_selectJoints failed to allocate hideSelectedJoints\n" NORMAL);
                                            free(mc->selectedJoints);
                                            mc->selectedJoints=0;
                                            return 0;
                                          } else
                                          {
                                             memset(mc->hideSelectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
                                          }

                                  }




  if ( (mc->selectedJoints!=0) && (mc->hideSelectedJoints!=0) )
  {
    unsigned int success=1;

    BVHJointID jID=0;
    fprintf(stderr,"Selecting : ");

    unsigned int i=0;
    for (i=iplus1+1; i<=iplus1+numberOfValues; i++)
     {
        if (
              //bvh_getJointIDFromJointName(mc,argv[i],&jID)
              bvh_getJointIDFromJointNameNocase(mc,argv[i],&jID)
           )
         {
           fprintf(stderr,GREEN "%s " NORMAL,argv[i]);

           mc->selectedJoints[jID]=1;
           mc->hideSelectedJoints[jID]=0;
           fprintf(stderr,"%u ",jID);

           if(includeEndSites)
                   {
                       if (mc->jointHierarchy[jID].hasEndSite)
                       {
                            fprintf(stderr,GREEN "EndSite_%s  " NORMAL,argv[i]);
                       }
                   }
           //-------------------------------------------------

         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");


    return success;
  }

  //We failed to allocate both so deallocate everything..
  mc->selectionIncludesEndSites=0;
  mc->numberOfJointsWeWantToSelect=0;
  if (mc->selectedJoints!=0) { free(mc->selectedJoints); mc->selectedJoints=0; }
  if (mc->hideSelectedJoints!=0) { free(mc->hideSelectedJoints); mc->hideSelectedJoints=0; }

  return 0;
}


int bvh_selectJointsToHide2D(
                             struct BVH_MotionCapture * mc,
                             unsigned int numberOfValues,
                             unsigned int includeEndSites,
                             const char **argv,
                             unsigned int iplus1
                            )
{
  if ( (mc->selectedJoints!=0) && (mc->hideSelectedJoints!=0) )
  {
    unsigned int success=1;

    unsigned int i=0;
    BVHJointID jID=0;
    fprintf(stderr,"Hiding 2D Coordinates : ");
    for (i=iplus1+1; i<=iplus1+numberOfValues; i++)
     {
      if (
              bvh_getJointIDFromJointNameNocase(mc,argv[i],&jID)
              //bvh_getJointIDFromJointName(mc,argv[i],&jID)
           )
         {
           if (jID<mc->jointHierarchySize)
           {
            //-------------------------------------------------
            fprintf(stderr,GREEN "%s " NORMAL,argv[i]);
            mc->hideSelectedJoints[jID]=1;
            fprintf(stderr,"%u ",jID);

            if(includeEndSites)
                   {
                     if(mc->selectionIncludesEndSites)
                          {
                              if (mc->jointHierarchy[jID].hasEndSite)
                                {
                                  fprintf(stderr,GREEN "EndSite_%s  " NORMAL,argv[i]);
                                }
                          }
                   } else
                   {
                    if (mc->jointHierarchy[jID].hasEndSite)
                                {
                                  fprintf(stderr,YELLOW "EndSite_%s(protected) " NORMAL,argv[i]);
                                  mc->hideSelectedJoints[jID]=2;
                                }
                   }
            //-------------------------------------------------
           } else
         {
           fprintf(stderr,RED "Joint retreived is erroneous.. " NORMAL);
           success=0;
         }
         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");
    return success;
  }

 fprintf(stderr,RED "bvh_selectJointsToHide2D cannot work without a prior bvh_selectJoints call..\n" NORMAL);
 return 0;
}











//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        PRINT STATE
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------





 int bvh_removeSelectedFrames(struct BVH_MotionCapture * bvhMotion,unsigned int * framesToRemove)
 {
   if ( (bvhMotion==0)||(framesToRemove==0) ) { return 0; }

   unsigned int offsetMID=0;
   unsigned int copyingEngaged=0;
   unsigned int framesEliminated=0;
   //---------------------------------------------------
   for (unsigned int fID=0; fID<bvhMotion->numberOfFrames; fID++)
   {
     if (framesToRemove[fID])
     {
        copyingEngaged=1;
        offsetMID+=bvhMotion->numberOfValuesPerFrame;
        ++framesEliminated;
     }

     if (copyingEngaged)
     {
       unsigned int mIDStart = fID * bvhMotion->numberOfValuesPerFrame;
       unsigned int mIDEnd   = (fID+1) * bvhMotion->numberOfValuesPerFrame;

       if (mIDEnd+offsetMID>=bvhMotion->numberOfFrames * bvhMotion->numberOfValuesPerFrame)
       {
          //Done erasing..
          break;
          //--------------
       }  else
       {
        for (unsigned int mID=mIDStart; mID<mIDEnd; mID++)
          {
            bvhMotion->motionValues[mID]=bvhMotion->motionValues[mID+offsetMID];
          }
       }
     }
   }
   //---------------------------------------------------

   bvhMotion->numberOfFramesEncountered=bvhMotion->numberOfFramesEncountered-framesEliminated;
   bvhMotion->numberOfFrames=bvhMotion->numberOfFrames-framesEliminated;

   return 1;
 }










//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        MOTION BUFFER
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void freeMotionBuffer(struct MotionBuffer ** pointerToMB)
{
 struct MotionBuffer *mb = *pointerToMB;
 if (mb!=0)
 {
  if(mb->motion!=0)
   {
     free(mb->motion);
     mb->motion=0;
   }
  free(mb);
  //Also clean the pointer..
  *pointerToMB=0;
 }
}
//---------------------------------------------------------

int copyMotionBuffer(struct MotionBuffer * dst,struct MotionBuffer * src)
{
  if ( (src!=0) && (dst!=0) )
  {
    if ( (src->motion!=0) && (dst->motion!=0) )
    {
     if (src->bufferSize == dst->bufferSize)
      {
       // 0.01
       memcpy(dst->motion,src->motion,sizeof(float) * src->bufferSize);
       // 0.16
       //for (unsigned int i=0; i<dst->bufferSize; i++) { dst->motion[i] = src->motion[i]; }
       return 1;
      }
      else
      {
       fprintf(stderr,RED "copyMotionBuffer: Buffer Size mismatch (Source %u/Destination %u)..\n" NORMAL,src->bufferSize,dst->bufferSize);
       return 0;
      }
     } //Src->Motion and Dst->Motion are ok
  } // Src and Dst are ok

  fprintf(stderr,RED "copyMotionBuffer: Failed due to incorrect allocations..\n" NORMAL);
  return 0;
}

struct MotionBuffer * mallocNewMotionBuffer(struct BVH_MotionCapture * mc)
{
  if (mc==0) { return 0; }
  //----------------------

  struct MotionBuffer * newBuffer = (struct MotionBuffer *)  malloc(sizeof(struct MotionBuffer));
  if (newBuffer!=0)
  {
    newBuffer->bufferSize = mc->numberOfValuesPerFrame;
    newBuffer->motion = (float *) malloc(sizeof(float) * newBuffer->bufferSize);
    if (newBuffer->motion!=0)
    {
      memset(newBuffer->motion,0,sizeof(float) * newBuffer->bufferSize);
    } else
    {
      //RollBack..!
      newBuffer->bufferSize=0;
      free(newBuffer);
      newBuffer=0;
    }
  }
  //----------------------
   return newBuffer;
}
//---------------------------------------------------------


struct MotionBuffer * mallocNewMotionBufferAndCopy(struct BVH_MotionCapture * mc,struct MotionBuffer * whatToCopy)
{
  if (mc==0) { return 0; }
  if (whatToCopy==0) { fprintf(stderr,"mallocNewMotionBufferAndCopy: Unable to copy from empty\n"); return 0; }
  if (whatToCopy->motion==0) { fprintf(stderr,"mallocNewMotionBufferAndCopy: Unable to copy from empty motion\n"); return 0; }
  //----------------------

  struct MotionBuffer * newBuffer = (struct MotionBuffer *)  malloc(sizeof(struct MotionBuffer));
  if (newBuffer!=0)
  {
    newBuffer->bufferSize = mc->numberOfValuesPerFrame;
    if (mc->numberOfValuesPerFrame != whatToCopy->bufferSize)
    {
        fprintf(stderr,RED "mallocNewMotionBufferAndCopy: Mismatching sizes %u vs %u\n" NORMAL,mc->numberOfValuesPerFrame,whatToCopy->bufferSize);
    }

    newBuffer->motion = (float *) malloc(sizeof(float) * newBuffer->bufferSize);
    if (newBuffer->motion!=0)
    {
      unsigned int numberOfItems = newBuffer->bufferSize;
      if (mc->numberOfValuesPerFrame>whatToCopy->bufferSize)
           { numberOfItems = newBuffer->bufferSize; }

      for (unsigned int i=0; i<numberOfItems; i++)
      {
        newBuffer->motion[i]=whatToCopy->motion[i];
      }
    } else
    {
      //RollBack..!
      newBuffer->bufferSize=0;
      free(newBuffer);
      newBuffer=0;
    }
  }

  //----------------------
  return newBuffer;
}
//---------------------------------------------------------





//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        PRINT STATE
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void bvh_printBVH(struct BVH_MotionCapture * bvhMotion)
{
  if (bvhMotion==0) {  fprintf(stdout,"\n\n\nCan't print empty BVH dataset..\n"); return; }
  fprintf(stdout,"\n\n\nPrinting BVH dataset..\n");

  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    fprintf(stdout,"___________________________________\n");
    fprintf(stdout,GREEN "Joint %u - %s/%s " NORMAL ,i,bvhMotion->jointHierarchy[i].jointName,bvhMotion->jointHierarchy[i].jointNameLowercase);
    unsigned int parentID = bvhMotion->jointHierarchy[i].parentJoint;
    fprintf(stdout," | Parent %u - %s \n",parentID,bvhMotion->jointHierarchy[parentID].jointName);
    //===============================================================
    if (bvhMotion->jointHierarchy[i].loadedChannels>0)
    {
     fprintf(stdout,"Has %u channels - ",bvhMotion->jointHierarchy[i].loadedChannels);
     if ( bvhMotion->jointHierarchy[i].channelRotationOrder==0 ) { fprintf(stdout,RED "!");}
     fprintf(stdout,"Rotation Order: %s \n" NORMAL,rotationOrderNames[(unsigned int) bvhMotion->jointHierarchy[i].channelRotationOrder]);
     for (unsigned int z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
      {
        unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
        fprintf(stdout,"%s ",channelNames[cT]);
      }
     fprintf(stdout,"\n");
    } else
    {
     fprintf(stdout,"Has no channels\n");
    }
    //===============================================================
    float distance =  bvhMotion->jointHierarchy[i].offset[0] * bvhMotion->jointHierarchy[i].offset[0];
          distance += bvhMotion->jointHierarchy[i].offset[1] * bvhMotion->jointHierarchy[i].offset[1];
          distance += bvhMotion->jointHierarchy[i].offset[2] * bvhMotion->jointHierarchy[i].offset[2];
     distance = sqrt(distance);
     fprintf(stdout,"Length : %0.2f\n",distance);
    //===============================================================
     fprintf(stdout,"Offset : ");
     for (unsigned int z=0; z<3; z++)
      {
        fprintf(stdout,"%0.5f ",bvhMotion->jointHierarchy[i].offset[z]);
      }
     fprintf(stdout,"\n");
    //===============================================================
    fprintf(stdout,"isRoot %u - ",(unsigned int) bvhMotion->jointHierarchy[i].isRoot);
    fprintf(stdout,"isEndSite %u - ",(unsigned int) bvhMotion->jointHierarchy[i].isEndSite);
    fprintf(stdout,"hasEndSite %u\n",(unsigned int) bvhMotion->jointHierarchy[i].hasEndSite);

    fprintf(stdout,"hasQuaternion %u - ",(unsigned int) bvhMotion->jointHierarchy[i].hasQuaternionRotation);
    fprintf(stdout,"hasRodrigues %u - ",(unsigned int) bvhMotion->jointHierarchy[i].hasRodriguesRotation);
    fprintf(stdout,"hasPosition %u\n",(unsigned int) bvhMotion->jointHierarchy[i].hasPositionalChannels);

    fprintf(stdout,"level %u\n",bvhMotion->jointHierarchy[i].hierarchyLevel );

    fprintf(stdout,"----------------------------------\n");
  }


  fprintf(stdout,"Motion data\n");
  fprintf(stdout,"___________________________________\n");
  fprintf(stdout,"Number of values per frame : %u \n",bvhMotion->numberOfValuesPerFrame);
  fprintf(stdout,"Loaded motion frames : %u \n",bvhMotion->numberOfFramesEncountered);
  fprintf(stdout,"Frame time : %0.8f \n",bvhMotion->frameTime);
  fprintf(stdout,"___________________________________\n");
}


void bvh_printBVHJointToMotionLookupTable(struct BVH_MotionCapture * bvhMotion)
{
  fprintf(stdout,"\n\n\nPrinting BVH JointToMotion lookup table..\n");
  fprintf(stdout,"_______________________________________________\n");
  for (unsigned int fID=0; fID<bvhMotion->numberOfFrames; fID++)
  {
   for (unsigned int jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
     for (unsigned int channelNumber=0; channelNumber<bvhMotion->jointHierarchy[jID].loadedChannels; channelNumber++ )
     {
         unsigned int channelTypeID = bvhMotion->jointHierarchy[jID].channelType[channelNumber];
         unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

         fprintf(stdout,"f[%u].%s.%s(%u)=%0.2f " ,
                 fID,
                 bvhMotion->jointHierarchy[jID].jointName,
                 channelNames[channelTypeID],
                 mID,
                 bvh_getMotionValue(bvhMotion,mID)
                 );
     }
    }
   fprintf(stdout,"\n\n");
  }
  fprintf(stdout,"_______________________________________________\n");
}

