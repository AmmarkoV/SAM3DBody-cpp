#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bvh_cut_paste.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

int scanJointHierarchyUntilThisGroupEnds(
                                         struct BVH_MotionCapture * mc,
                                         BVHJointID jID,
                                         BVHJointID * jIDLastGroupJoint,
                                         unsigned int * numberOfChannelsContained
                                        )
{
  fprintf(stderr,"scanJointHierarchyUntilThisGroupEnds(%s): contains ",mc->jointHierarchy[jID].jointName);

  *jIDLastGroupJoint=jID;
  unsigned int targetHierarchyLevel=mc->jointHierarchy[jID].hierarchyLevel;
  *numberOfChannelsContained = mc->jointHierarchy[jID].loadedChannels;

  ++jID;//Start from the next joint..
  while (jID<mc->numberOfFrames)
  {
    if (targetHierarchyLevel>=mc->jointHierarchy[jID].hierarchyLevel)
    {
      fprintf(stderr,"all of them are %u \n",*numberOfChannelsContained);
      //We have reached the end..!
      *jIDLastGroupJoint=jID;
      return 1;
    } else
    {
      *numberOfChannelsContained+=mc->jointHierarchy[jID].loadedChannels;
      fprintf(stderr,"(%s-%u) ",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].loadedChannels);
      jID++;
    }
  }

 fprintf(stderr,"\n");
 return 0;
}



int checkIfJointsHaveSameGraphOutline(
                                       struct BVH_MotionCapture * mc,
                                       BVHJointID jIDA,
                                       BVHJointID jIDB,
                                       unsigned int * rangeOfJIDA,
                                       unsigned int * rangeOfJIDB,
                                       unsigned int * numberOfChannelsContainedJIDA,
                                       unsigned int * numberOfChannelsContainedJIDB
                                     )
{
  //We assume jIDA is before jIDB if now we fix it
  if (jIDA>jIDB)
  {
    BVHJointID tmp=jIDA;
    jIDA=jIDB;
    jIDB=tmp;
  }
  //----------------------------

  BVHJointID jIDALastJoint,jIDBLastJoint;

  //We see the range of each of the joint graphs
  if (
      (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jIDA,
                                              &jIDALastJoint,
                                              numberOfChannelsContainedJIDA
                                            )
      ) &&
      (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jIDB,
                                              &jIDBLastJoint,
                                              numberOfChannelsContainedJIDB
                                            )
      )
     )
     {
       unsigned int rA = (jIDALastJoint-jIDA);
       unsigned int rB = (jIDBLastJoint-jIDB);

       //If the last 2 arguments are not set to null we use them..!
       if (rangeOfJIDA!=0) { *rangeOfJIDA = rA; }
       if (rangeOfJIDB!=0) { *rangeOfJIDB = rB; }

       //If they are the same we can copy paste one to the other
       return ((rA==rB) && (*numberOfChannelsContainedJIDA==*numberOfChannelsContainedJIDB));
     }

  return 0;
}


float * allocateBufferThatCanContainJointAndChildren(
                                                      struct BVH_MotionCapture * mc,
                                                      BVHJointID jID
                                                    )
{
  unsigned int numberOfChannels=0;
  BVHJointID jIDLastJoint;
  if  (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jID,
                                              &jIDLastJoint,
                                              &numberOfChannels
                                            )
      )
      {
        return (float *) malloc(sizeof(float) * (numberOfChannels));
      }
  return 0;
}



int copyJointAndChildrenToBuffer(
                                 struct BVH_MotionCapture * mc,
                                 float * buffer,
                                 BVHJointID jID,
                                 unsigned int rangeNumber,
                                 BVHFrameID  fID
                                )
{
  unsigned int mID = (fID * mc->numberOfValuesPerFrame) + mc->jointToMotionLookup[jID].jointMotionOffset;
  memcpy(
         buffer ,
         &mc->motionValues[mID],
         rangeNumber * sizeof(float)
        );

 return 1;
}

int copyBufferToJointAndChildren(
                                 struct BVH_MotionCapture * mc,
                                 float * buffer,
                                 BVHJointID jID,
                                 unsigned int rangeNumber,
                                 BVHFrameID  fID
                                )
{
  unsigned int mID = (fID * mc->numberOfValuesPerFrame) + mc->jointToMotionLookup[jID].jointMotionOffset;
  memcpy(
         &mc->motionValues[mID],
         buffer ,
         rangeNumber * sizeof(float)
        );

 return 1;
}



int bvh_EraseAndAllocateSpaceForNumberOfFrames(struct BVH_MotionCapture * mc,unsigned int targetNumberOfFrames)
{
  if (mc==0) { return 0; }

  //Erased
  if ( mc->motionValues !=0 )
  {
    free(mc->motionValues);
    mc->motionValues=0;
    mc->numberOfFrames=0;
    mc->numberOfFramesEncountered=0;
    mc->motionValuesSize=0;
  }

  mc->motionValues  = (float*) malloc(sizeof(float) * mc->numberOfValuesPerFrame * targetNumberOfFrames );
  if (mc->motionValues!=0)
  {
    //Also erased new motion value block
    memset(mc->motionValues,0,sizeof(float) * targetNumberOfFrames *  mc->numberOfValuesPerFrame);
    //Record keeping for the new buffer
    mc->numberOfFrames=targetNumberOfFrames;
    mc->numberOfFramesEncountered=0;
    mc->motionValuesSize=mc->numberOfValuesPerFrame* mc->numberOfFrames;
    return 1;
  }

 return 0;
}



int bvh_GrowMocapFileByCopyingOtherMocapFile(
                                              struct BVH_MotionCapture * mc,
                                              struct BVH_MotionCapture * mcSource
                                             )
{
  if (mc==0)                           { fprintf(stderr,"No motion capture file to repeat\n");  return 0; }
  if (mc->motionValues==0)             { fprintf(stderr,"No data to repeat\n");                 return 0; }
  if (mc->motionValuesSize==0)         { fprintf(stderr,"Data to repeat has zero data size\n"); return 0; }
  if (mcSource==0)                     { fprintf(stderr,"No motion capture file to repeat\n");  return 0; }
  if (mcSource->motionValues==0)       { fprintf(stderr,"No data to repeat\n");                 return 0; }
  if (mcSource->motionValuesSize==0)   { fprintf(stderr,"Data to repeat has zero data size\n"); return 0; }
  //-------------------------------------------------------------------------------------------------
  fprintf(stderr,"Asked to copy %s to %s\n",mc->fileName,mcSource->fileName);
  fprintf(stderr,"Will now try to allocate %lu KB of memory\n",(sizeof(float) * (mc->motionValuesSize+ mcSource->motionValuesSize)) / 1024);

  if (mc->jointHierarchySize!=mcSource->jointHierarchySize)
  {
      fprintf(stderr,"Inconsistent Source/Target Files..\n");
      return 0;
  }

  if (mc->numberOfValuesPerFrame!=mcSource->numberOfValuesPerFrame)
  {
      fprintf(stderr,"Inconsistent Source/Target Files..\n");
      return 0;
  }

  //The new memory size will be the added memory size!
  unsigned int newMCSize = sizeof(float) * ( mc->motionValuesSize + mcSource->motionValuesSize);
  float * newMotionValues = (float*) malloc(newMCSize);
  if (newMotionValues==0) { fprintf(stderr,"Could not allocate new motion values\n"); return 0; }

  //Make sure new motion values are set to 0 to keep things tidy
  memset(newMotionValues,0,newMCSize);

  float * oldMotionValues = mc->motionValues;
  float * ptr=newMotionValues;

  fprintf(stderr,"Copying : ");
  //Copying original data
  memcpy(ptr,mc->motionValues,sizeof(float) * mc->motionValuesSize);
  ptr+=mc->motionValuesSize;
  //Copy extra data
  memcpy(ptr,mcSource->motionValues,sizeof(float) * mcSource->motionValuesSize);
  ptr+=mcSource->motionValuesSize;
  fprintf(stderr," Done \n");
  //fprintf(stderr," (offset %u/%u) \n",newMotionValues-ptr,newMCSize);

  mc->numberOfFrames            += mcSource->numberOfFrames;
  mc->numberOfFramesEncountered += mcSource->numberOfFrames;
  mc->motionValuesSize          += mcSource->motionValuesSize;
  mc->motionValues               = newMotionValues;
  free(oldMotionValues);

 return 1;
}



int bvh_GrowMocapFileByCopyingExistingMotions(
                                              struct BVH_MotionCapture * mc,
                                              unsigned int timesToRepeat
                                             )
{
  if (timesToRepeat==0)          { fprintf(stderr,"No times to repeat\n");                return 1; }
  if (mc==0)                     { fprintf(stderr,"No motion capture file to repeat\n");  return 0; }
  if (mc->motionValues==0)       { fprintf(stderr,"No data to repeat\n");                 return 0; }
  if (mc->motionValuesSize==0)   { fprintf(stderr,"Data to repeat has zero data size\n"); return 0; }
  //-------------------------------------------------------------------------------------------------
  fprintf(stderr,"Asked to repeat %u times the %u existing motion records\n",timesToRepeat,mc->numberOfFramesEncountered);
  fprintf(stderr,"Will now try to allocate %lu KB of memory\n",(sizeof(float) * mc->motionValuesSize * (timesToRepeat+1)) / 1024);

  float * newMotionValues = (float*) malloc(sizeof(float) * mc->motionValuesSize * (timesToRepeat+1) );
  if (newMotionValues==0) { fprintf(stderr,"Could not allocate new motion values\n"); return 0; }

  float * oldMotionValues = mc->motionValues;
  float * ptr=newMotionValues;

  fprintf(stderr,"Copying : ");
  unsigned int r=0;
  for (r=0; r<timesToRepeat+1; r++)
  {
    memcpy(ptr,oldMotionValues,sizeof(float) * mc->motionValuesSize);
    ptr+=mc->motionValuesSize;
    if (r%50000==0) { fprintf(stderr,"."); }
  }
  fprintf(stderr," Done\n");

 mc->numberOfFrames+=mc->numberOfFrames*timesToRepeat;
 mc->numberOfFramesEncountered+=mc->numberOfFrames;
 mc->motionValuesSize+=mc->motionValuesSize*timesToRepeat;
 mc->motionValues = newMotionValues;
 free(oldMotionValues);

 return 1;
}



int bvh_GrowMocapFileByGeneratingPoseFromAllViewingAngles(
                                                          struct BVH_MotionCapture * mc,
                                                          unsigned int poseNumber
                                                         )
{
    if (mc==0) { return 0; }
    if (mc->motionValues==0) { return 0; }
    if (poseNumber>=mc->numberOfFramesEncountered) { return 0; }

    fprintf(stderr,"bvh_GrowMocapFileByGeneratingPoseFromAllViewingAngles(mc,%u)\n",poseNumber);

    float * valuesThatWillBeCopied = (float * ) malloc(sizeof(float) * mc->numberOfValuesPerFrame);
    if (valuesThatWillBeCopied!=0)
    {
      memcpy(
             valuesThatWillBeCopied,
             &mc->motionValues[poseNumber * mc->numberOfValuesPerFrame],
             mc->numberOfValuesPerFrame * sizeof(float)
            );

      valuesThatWillBeCopied[0]=0.0;
      valuesThatWillBeCopied[1]=0.0;

      if ( mc->numberOfFrames <360 )
      {
         float * newMotionValues = (float*) malloc(sizeof(float) * mc->numberOfValuesPerFrame * (360) );
         if (newMotionValues!=0)
         {
           free(mc->motionValues);
           mc->motionValues=newMotionValues;
           mc->numberOfFrames=360;
           mc->numberOfFramesEncountered=mc->numberOfFrames;
           mc->motionValuesSize=mc->numberOfValuesPerFrame* mc->numberOfFrames;
         } else
         {
            free(valuesThatWillBeCopied);
           fprintf(stderr,"Failed to allocate memory for new motion values\n");
           return 0;
         }
      }



      if ( mc->numberOfFrames >= 360 )
      {
         for (int fID=0; fID<360; fID++)
          {
              //Change Y coordinate only
              valuesThatWillBeCopied[4]=(float) fID;

             memcpy(
                     &mc->motionValues[fID * mc->numberOfValuesPerFrame],
                     valuesThatWillBeCopied,
                     mc->numberOfValuesPerFrame * sizeof(float)
                    );


          }
          mc->numberOfFramesEncountered=360;
          mc->numberOfFrames=360;
      }

      free(valuesThatWillBeCopied);
      return 1;
    }
   return 0;
}




int bvh_GrowMocapFileBySwappingJointAndItsChildren(
                                                     struct BVH_MotionCapture * mc,
                                                     const char * jointNameA,
                                                     const char * jointNameB,
                                                     int alsoIncludeOriginalMotion
                                                   )
{
  BVHJointID jIDA,jIDB;
  unsigned int rangeOfJIDA,rangeOfJIDB;
  unsigned int numberOfChannelsContainedJIDA,numberOfChannelsContainedJIDB;

  if ( (bvh_getJointIDFromJointNameNocase(mc,jointNameA,&jIDA)) && (bvh_getJointIDFromJointNameNocase(mc,jointNameB,&jIDB)) )
  {
   fprintf(stderr,"We have resolved %s to %u and %s to %u\n",jointNameA,jIDA,jointNameB,jIDB);
   if (
        checkIfJointsHaveSameGraphOutline(
                                          mc,
                                          jIDA,
                                          jIDB,
                                          &rangeOfJIDA,
                                          &rangeOfJIDB,
                                          &numberOfChannelsContainedJIDA,
                                          &numberOfChannelsContainedJIDB
                                         )
      )
    {
     fprintf(stderr,"bvh_GrowMocapFileBySwappingJointAndItsChildren");
     unsigned int initialNumberOfFrames = mc->numberOfFrames;
     fprintf(stderr,"Initially had %u frames\n",initialNumberOfFrames);

     //----------------------------------------------------------------
     if (alsoIncludeOriginalMotion)
     {
      fprintf(stderr,"And we where asked to double them\n");
      if (
          bvh_GrowMocapFileByCopyingExistingMotions(mc,1)
         )
      {
          //Successfully Grew buffer to alsoIncludeOriginalMotion
      } else
      {
        fprintf(stderr,"Could not grow our movement buffer to facilitate swapping\n");
        return 0;
      }
     }
     //----------------------------------------------------------------


       fprintf(stderr,"We now have %u frames\n",mc->numberOfFrames);
       float * temporaryMotionBufferA = allocateBufferThatCanContainJointAndChildren( mc, jIDA );
       float * temporaryMotionBufferB = allocateBufferThatCanContainJointAndChildren( mc, jIDB );
       if ( (temporaryMotionBufferA!=0) && (temporaryMotionBufferB!=0) )
       {
        BVHFrameID fID=0;
        for (fID=0; fID<initialNumberOfFrames; fID++)
         {
            if (
                (copyJointAndChildrenToBuffer(mc,temporaryMotionBufferA,jIDA,numberOfChannelsContainedJIDA,fID)) &&
                (copyJointAndChildrenToBuffer(mc,temporaryMotionBufferB,jIDB,numberOfChannelsContainedJIDB,fID))
               )
                {
                  if (
                      (copyBufferToJointAndChildren(mc,temporaryMotionBufferB,jIDA,numberOfChannelsContainedJIDA,fID)) &&
                      (copyBufferToJointAndChildren(mc,temporaryMotionBufferA,jIDB,numberOfChannelsContainedJIDB,fID))
                     )
                     {
                      //Success for mID
                     } else
                     { fprintf(stderr,"Error accessing and copying buffers back to joints %s/%s at frame %u\n",jointNameA,jointNameB,fID); }
                } else
                { fprintf(stderr,"Error accessing and copying joints %s/%s at frame %u to buffers\n",jointNameA,jointNameB,fID); }
         }
        free(temporaryMotionBufferA);
        free(temporaryMotionBufferB);
        return 1;
       } else
       { fprintf(stderr,"Could not allocate temporary buffer\n");  }
    } else
    { fprintf(stderr,"Joints %s and %s do not have the same hierarchy graph outline and therefore cannot be swapped\n",jointNameA,jointNameB); }
  } else
  {
    fprintf(stderr,"Could not resolve %s and %s , maybe they got internally renamed?\n",jointNameA,jointNameB);
    fprintf(stderr,"Full list of joints is : \n");
    unsigned int jID=0;
     for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        fprintf(stderr,"   joint %u = %s\n",jID,mc->jointHierarchy[jID].jointName);
      }
  }

 fprintf(stderr,"Errors occured during bvh_GrowMocapFileByMirroringJointAndItsChildren\n");
 return 0;
}




int bvh_swapJointMotionsForFrameID(
                                    struct BVH_MotionCapture * mc,
                                    BVHFrameID fID,
                                    BVHJointID jIDA,
                                    BVHJointID jIDB,
                                    char flipX,char flipY,char flipZ
                                  )
{
  if (
       (!mc->jointHierarchy[jIDA].isEndSite) &&
       (!mc->jointHierarchy[jIDB].isEndSite)
     )
  {
  //-------------------------------------------------------
  float jIDA_vX = bvh_getJointRotationXAtFrame(mc,jIDA,fID);
  float jIDA_vY = bvh_getJointRotationYAtFrame(mc,jIDA,fID);
  float jIDA_vZ = bvh_getJointRotationZAtFrame(mc,jIDA,fID);
  //-------------------------------------------------------
  float jIDB_vX = bvh_getJointRotationXAtFrame(mc,jIDB,fID);
  float jIDB_vY = bvh_getJointRotationYAtFrame(mc,jIDB,fID);
  float jIDB_vZ = bvh_getJointRotationZAtFrame(mc,jIDB,fID);
  //-------------------------------------------------------
  float fX = 1.0,fY = 1.0,fZ = 1.0;
  //-------------------------------------------------------
  if (flipX) { fX=-1.0; }
  if (flipY) { fY=-1.0; }
  if (flipZ) { fZ=-1.0; }
  //-------------------------------------------------------
  bvh_setJointRotationXAtFrame(mc,jIDA,fID,fX*jIDB_vX);
  bvh_setJointRotationYAtFrame(mc,jIDA,fID,fY*jIDB_vY);
  bvh_setJointRotationZAtFrame(mc,jIDA,fID,fZ*jIDB_vZ);
  //-------------------------------------------------------
  bvh_setJointRotationXAtFrame(mc,jIDB,fID,fX*jIDA_vX);
  bvh_setJointRotationYAtFrame(mc,jIDB,fID,fY*jIDA_vY);
  bvh_setJointRotationZAtFrame(mc,jIDB,fID,fZ*jIDA_vZ);
  //-------------------------------------------------------
  }
  return 1;
}



int bvh_symmetricJointNameParser(struct BVH_MotionCapture * mc)
{
  if (!mc->checkedForSymmetricJoints)
  {
  //BVHJointID jIDA,jIDB;
  //unsigned int rangeOfJIDA,rangeOfJIDB;
  //unsigned int numberOfChannelsContainedJIDA,numberOfChannelsContainedJIDB;
  //-------------------------
  BVHJointID symmetricJID=0;
  char symmetricTo[512]={0};
  //-------------------------
  //bvh_printBVH(mc);
  for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
  {
    unsigned int jNameLength = strlen(mc->jointHierarchy[jID].jointNameLowercase);
    if (jNameLength>2)
    {
        if ( (mc->jointHierarchy[jID].jointNameLowercase[jNameLength-2]=='.') &&
             (mc->jointHierarchy[jID].jointNameLowercase[jNameLength-1]=='l') )
             {
                 //This is a left symmetric joint!
                 snprintf(symmetricTo,512,"%s",mc->jointHierarchy[jID].jointNameLowercase);
                 symmetricTo[jNameLength-1]='r';
                 if (bvh_getJointIDFromJointNameNocase(mc,symmetricTo,&symmetricJID))
                 {
                    fprintf(stderr,RED "Joint %s is Left Symmetric to %s\n" NORMAL,mc->jointHierarchy[jID].jointNameLowercase,symmetricTo);
                    mc->jointHierarchy[jID].symmetricJoint=symmetricJID;
                    mc->jointHierarchy[jID].symmetryIsLeftJoint=1;
                    mc->jointHierarchy[symmetricJID].symmetricJoint=jID;
                    mc->jointHierarchy[symmetricJID].symmetryIsLeftJoint=0;
                 }  else
                 { fprintf(stderr,RED "Could not resolve joint `%s` \n",symmetricTo); }
             } else
        if ( (mc->jointHierarchy[jID].jointNameLowercase[jNameLength-2]=='.') &&
             (mc->jointHierarchy[jID].jointNameLowercase[jNameLength-1]=='r') )
             {
                 //This is a right symmetric joint!
                 snprintf(symmetricTo,512,"%s",mc->jointHierarchy[jID].jointNameLowercase);
                 symmetricTo[jNameLength-1]='l';
                 if (bvh_getJointIDFromJointNameNocase(mc,symmetricTo,&symmetricJID))
                 {
                    fprintf(stderr,GREEN "Joint %s is Right Symmetric to %s\n" NORMAL,mc->jointHierarchy[jID].jointNameLowercase,symmetricTo);
                    mc->jointHierarchy[jID].symmetricJoint=symmetricJID;
                    mc->jointHierarchy[jID].symmetryIsLeftJoint=1;
                    mc->jointHierarchy[symmetricJID].symmetricJoint=jID;
                    mc->jointHierarchy[symmetricJID].symmetryIsLeftJoint=0;
                 }  else
                 { fprintf(stderr,RED "Could not resolve joint `%s` \n",symmetricTo); }
             } else
        if ( (mc->jointHierarchy[jID].jointNameLowercase[0]=='l')  )
             {
                 //This is a left symmetric joint!
                 snprintf(symmetricTo,512,"%s",mc->jointHierarchy[jID].jointNameLowercase);
                 symmetricTo[0]='r';
                 if (bvh_getJointIDFromJointNameNocase(mc,symmetricTo,&symmetricJID))
                 {
                    fprintf(stderr,RED "Joint %s is Left Symmetric to %s\n" NORMAL,mc->jointHierarchy[jID].jointNameLowercase,symmetricTo);
                    mc->jointHierarchy[jID].symmetricJoint=symmetricJID;
                    mc->jointHierarchy[jID].symmetryIsLeftJoint=1;
                    mc->jointHierarchy[symmetricJID].symmetricJoint=jID;
                    mc->jointHierarchy[symmetricJID].symmetryIsLeftJoint=0;
                 }  else
                 { fprintf(stderr,RED "Could not resolve joint `%s` \n",symmetricTo); }
             } else
        if ( (mc->jointHierarchy[jID].jointNameLowercase[0]=='r')  )
             {
                 //This is a right symmetric joint!
                 snprintf(symmetricTo,512,"%s",mc->jointHierarchy[jID].jointNameLowercase);
                 symmetricTo[0]='l';
                 if (bvh_getJointIDFromJointNameNocase(mc,symmetricTo,&symmetricJID))
                 {
                    fprintf(stderr,GREEN "Joint %s is Right Symmetric to %s\n" NORMAL,mc->jointHierarchy[jID].jointNameLowercase,symmetricTo);
                    mc->jointHierarchy[jID].symmetricJoint=symmetricJID;
                    mc->jointHierarchy[jID].symmetryIsLeftJoint=1;
                    mc->jointHierarchy[symmetricJID].symmetricJoint=jID;
                    mc->jointHierarchy[symmetricJID].symmetryIsLeftJoint=0;
                 }  else
                 { fprintf(stderr,RED "Could not resolve joint `%s` \n",symmetricTo); }
             }
    }// nameLength>2
  } //for


    mc->checkedForSymmetricJoints = 1;
  }
 return 1;
}




int bvh_symmetricflipLeftAndRight(
                                  struct BVH_MotionCapture * mc,
                                  BVHFrameID fID
                                 )
{
    if (bvh_symmetricJointNameParser(mc))
    {
      //-------------------------------------------------------
      BVHJointID jIDA=0;
      //-------------------------------------------------------
      bvh_setJointPositionXAtFrame(mc,mc->rootJointID,fID,-1.0*bvh_getJointPositionXAtFrame(mc,mc->rootJointID,fID));
      //-------------------------------------------------------
      float root_vX = bvh_getJointRotationXAtFrame(mc,mc->rootJointID,fID);
      float root_vY = bvh_getJointRotationYAtFrame(mc,mc->rootJointID,fID);
      float root_vZ = bvh_getJointRotationZAtFrame(mc,mc->rootJointID,fID);
      //-------------------------------------------------------
      float fX = 1.0,fY = -1.0,fZ = -1.0;
      //-------------------------------------------------------
      bvh_setJointRotationXAtFrame(mc,jIDA,fID,fX*root_vX);
      bvh_setJointRotationYAtFrame(mc,jIDA,fID,fY*root_vY);
      bvh_setJointRotationZAtFrame(mc,jIDA,fID,fZ*root_vZ);
      //-------------------------------------------------------
      for (jIDA=0; jIDA<mc->jointHierarchySize; jIDA++)
        {
            if (mc->jointHierarchy[jIDA].symmetryIsLeftJoint)
            {
               BVHJointID jIDB = mc->jointHierarchy[jIDA].symmetricJoint;
               bvh_swapJointMotionsForFrameID(
                                               mc,
                                               fID,
                                               jIDA,
                                               jIDB,
                                               0,1,1
                                             );
            }
        }
        return 1;
    }
 return 0;
}
