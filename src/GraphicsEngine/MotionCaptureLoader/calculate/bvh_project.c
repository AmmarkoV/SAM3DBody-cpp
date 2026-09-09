#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../bvh_loader.h"
#include "bvh_project.h"

void bvh_cleanTransform(
                        struct BVH_MotionCapture * mc,
                        struct BVH_Transform     * bvhTransform
                       )
{
 if ( bvh_allocateTransform(mc,bvhTransform) )
 {
  bvhTransform->jointsOccludedIn2DProjection=0;
  bvhTransform->torso.exists=0;

  if (bvhTransform->numberOfJointsSpaceAllocated < mc->jointHierarchySize )
  {
      fprintf(stderr,"bvh_cleanTransform problem with allocated transform\n");
      exit(1);
  }

  unsigned int jID=0;
  for (jID=0; jID<mc->jointHierarchySize; jID++)
         {
          bvhTransform->joint[jID].pos3D[0]=0.0;
          bvhTransform->joint[jID].pos3D[1]=0.0;
          bvhTransform->joint[jID].pos3D[2]=0.0;

          bvhTransform->joint[jID].pos2DCalculated=0;
          bvhTransform->joint[jID].isBehindCamera=0;

          bvhTransform->joint[jID].pos2D[0]=0.0;
          bvhTransform->joint[jID].pos2D[1]=0.0;
         }
 }
}



int bvh_projectTo2DHandleOcclusions(
                                    struct BVH_MotionCapture * mc,
                                    struct BVH_Transform     * bvhTransform
                                   )
{
  unsigned int jID;
       //First set all points as unoccluded..!
       for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         bvhTransform->joint[jID].isOccluded=0;
       }

       //CONTROLS HERE..
       #define USE_JOINT_TO_JOINT_OCCLUSIONS 0
       #define USE_TORSO_OCCLUSIONS 1
       #define DEBUG_TORSO_OCCLUSIONS 0
       #define OCCLUSION_THRESHOLD 6 // pixels
       //bvhTransform->joint[jID].isOccluded=0;

       #if DEBUG_TORSO_OCCLUSIONS
        fprintf(stderr,"Occlusion checking..\n");
       #endif // DEBUG_TORSO_OCCLUSIONS


       for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
        if (bvhTransform->joint[jID].pos2DCalculated)
        {

         #if USE_TORSO_OCCLUSIONS
         //If we have a torso ----------------------------------
         if ((bvhTransform->torso.exists) && (!mc->jointHierarchy[jID].isImmuneToTorsoOcclusions))
         {
           //If the point is behind torso (which is a fast check) ----------------------------------
           if (bvhTransform->joint[jID].pos3D[2]<bvhTransform->torso.averageDepth)
           {
            //If the point is inside the 2D rectangle of the torso ----------------------------------
            if (
                 (bvhTransform->torso.rectangle2D.x < bvhTransform->joint[jID].pos2D[0]) &&
                 (bvhTransform->joint[jID].pos2D[0] < bvhTransform->torso.rectangle2D.x + bvhTransform->torso.rectangle2D.width ) &&
                 (bvhTransform->torso.rectangle2D.y < bvhTransform->joint[jID].pos2D[1]) &&
                 (bvhTransform->joint[jID].pos2D[1] < bvhTransform->torso.rectangle2D.y + bvhTransform->torso.rectangle2D.height )
              )
            {
             bvhTransform->joint[jID].isOccluded=1;
             ++bvhTransform->jointsOccludedIn2DProjection;

             #if DEBUG_TORSO_OCCLUSIONS
             fprintf(stderr,"TORSO OCCLUSION\n");
             fprintf(
                     stderr,"Point @ %0.2f,%0.2f -> box %0.2f,%0.2f (%0.2fx%0.2f)\n",
                     bvhTransform->joint[jID].pos2D[0],
                     bvhTransform->joint[jID].pos2D[1],
                     bvhTransform->torso.rectangle2D.x,
                     bvhTransform->torso.rectangle2D.y,
                     bvhTransform->torso.rectangle2D.width,
                     bvhTransform->torso.rectangle2D.height
                    );
              #endif // DEBUG_TORSO_OCCLUSIONS

            }
            //------

           }
           #if DEBUG_TORSO_OCCLUSIONS
            else
           {
               fprintf(
                       stderr,"Point %s is behind torso (%0.2f < %0.2f )\n",
                        mc->jointHierarchy[jID].jointName,
                        bvhTransform->joint[jID].pos3D[2],
                        bvhTransform->torso.averageDepth
                       );
           }
           #endif // DEBUG_TORSO_OCCLUSIONS

          //------
         }
         #endif // USE_TORSO_OCCLUSIONS

         #if USE_JOINT_TO_JOINT_OCCLUSIONS
         //If the joint is still not occluded then do extra checks, otherwise conserve our CPU
         if (!bvhTransform->joint[jID].isOccluded)
         {
         //Check Joint for occlusion with all other joints that may be in front of it
         //------------------------------------------------------------------------------------------
         for (unsigned int  jID2=0; jID2<mc->jointHierarchySize; jID2++)
          {
           if ( (mc->selectedJoints==0) || (mc->selectedJoints[jID2]) )
           {
            if ( (jID2!=jID)  )
            {
             if (bvhTransform->joint[jID2].pos2DCalculated)
             {
               //We check if these two joints are very close together
               float diffA=bvhTransform->joint[jID].pos2D[0]-bvhTransform->joint[jID2].pos2D[0];
               float diffB=bvhTransform->joint[jID].pos2D[1]-bvhTransform->joint[jID2].pos2D[1];
               float distance=sqrt((diffA*diffA)+(diffB*diffB));
               if (distance<OCCLUSION_THRESHOLD)
                {
                  //If they are close together and joint jID2 is in front of jID
                  if (bvhTransform->joint[jID].pos3D[2]>bvhTransform->joint[jID2].pos3D[2])
                  {
                    //fprintf(stderr," %0.2f",distance);
                    //then jID is occluded..!
                    bvhTransform->joint[jID].isOccluded=1;
                    ++bvhTransform->jointsOccludedIn2DProjection;
                  } //If they have  the correct order
                } //If it is closer than the minimum distance required by OCCLUSION_THRESHOLD
             } //Only if we have 2D position calculated
           } //Excluding occlusions with itself
          } //Only occlude with selected joints, ignore all else
          } // Compare to all other joints
          //------------------------------------------------------------------------------------------
         }//If joint not already occluded
         #endif // USE_JOINT_TO_JOINT_OCCLUSIONS

        }
       }
     //------------------------------------------------------------------------------------------

  return 1;
}





int bvh_projectJIDTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     BVHJointID jID,
                     unsigned int               occlusions,
                     unsigned int               directRendering
                   )
{
      int res = 0;
      float position2D[3]={0.0,0.0,0.0};


      if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
      {
           float pos3DFloat[4];
           pos3DFloat[0]= (float) bvhTransform->joint[jID].pos3D[0];
           pos3DFloat[1]= (float) bvhTransform->joint[jID].pos3D[1];
           pos3DFloat[2]= (float) bvhTransform->joint[jID].pos3D[2];
           pos3DFloat[3]= (float) bvhTransform->joint[jID].pos3D[3];

          //Two cases here , we either want to want to render using our camera position where we have to do some extra matrix operations
          //to recenter our mesh and transform it accordingly
          //-------------------------------------------------------------------------------------------
          if (!directRendering)
          {
           float pos3DCenterFloat[4];
           pos3DCenterFloat[0]= (float) bvhTransform->centerPosition[0];
           pos3DCenterFloat[1]= (float) bvhTransform->centerPosition[1];
           pos3DCenterFloat[2]= (float) bvhTransform->centerPosition[2];
           pos3DCenterFloat[3]=1.0;

           simpleRendererRenderEx(
                                 renderer ,
                                 pos3DFloat,
                                 pos3DCenterFloat,
                                 0,
                                 0,
                                 &position2D[0],
                                 &position2D[1],
                                 &position2D[2],
                                 0 //<- we skip this !
                               );

         }
           else
         {
         //-------------------------------------------------------------------------------------------
         //Or we directly render using the matrices in our simplerenderer
         //-------------------------------------------------------------------------------------------
           fprintf(stderr, "DIRECT RENDER ");
           simpleRendererRenderUsingPrecalculatedMatrices(
                                                          renderer,
                                                          pos3DFloat,
                                                          &position2D[0],
                                                          &position2D[1],
                                                          &position2D[2]
                                                         );
          }
         //-------------------------------------------------------------------------------------------

           if (position2D[2]>=0.0)
           {
              bvhTransform->joint[jID].pos2D[0] = (float) position2D[0];
              bvhTransform->joint[jID].pos2D[1] = (float) position2D[1];
              bvhTransform->joint[jID].pos2DCalculated=1;
           } else
           {
              bvhTransform->joint[jID].pos2D[0] = 0.0;
              bvhTransform->joint[jID].pos2D[1] = 0.0;
              bvhTransform->joint[jID].isBehindCamera=1;
           }

           res = 1;
        }

 return res;
}








int bvh_projectTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     unsigned int               occlusions,
                     unsigned int               directRendering
                   )
{
      if (!bvhTransform) { return 0; }
      if (!bvhTransform->transformStructInitialized) { return 0; }
      if (!renderer) { return 0; }


      bvhTransform->jointsOccludedIn2DProjection=0;
      float position2D[3]={0.0,0.0,0.0};
      float pos3DFloat[4];

      //Do this once..!
      simpleRendererUpdateMovelViewTransform(renderer);

      //Then project 3D positions on 2D frame and save results..
      #if USE_TRANSFORM_HASHING
      fprintf(stderr,"bvh_projectTo2D called with %u items in hash\n",bvhTransform->lengthOfListOfJointIDsToTransform);
      for (unsigned int hashID=0; hashID<bvhTransform->lengthOfListOfJointIDsToTransform; hashID++)
         {
           //USING_HASH
           unsigned int jID=bvhTransform->listOfJointIDsToTransform[hashID];
      #else
      for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
         {
      #endif
        if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
        {
           pos3DFloat[0]= (float) bvhTransform->joint[jID].pos3D[0];
           pos3DFloat[1]= (float) bvhTransform->joint[jID].pos3D[1];
           pos3DFloat[2]= (float) bvhTransform->joint[jID].pos3D[2];
           pos3DFloat[3]= (float) bvhTransform->joint[jID].pos3D[3];
          //Two cases here , we either want to want to render using our camera position where we have to do some extra matrix operations
          //to recenter our mesh and transform it accordingly
          //-------------------------------------------------------------------------------------------
          if (!directRendering)
          {
           float pos3DCenterFloat[4] = {
                                         (float) bvhTransform->centerPosition[0],
                                         (float) bvhTransform->centerPosition[1],
                                         (float) bvhTransform->centerPosition[2],
                                         1.0
                                       };
           simpleRendererRenderEx(
                                  renderer ,
                                  pos3DFloat,
                                  pos3DCenterFloat,
                                  0,
                                  0,
                                  &position2D[0],
                                  &position2D[1],
                                  &position2D[2],
                                  0 //Don't update more than once per BVH skeleton
                                );

            //Should this only happen when position2DW>=0.0
            //bvhTransform->joint[jID].pos3D[0] = (float) pos3DCenterFloat[0];
            //bvhTransform->joint[jID].pos3D[1] = (float) pos3DCenterFloat[1];
            //bvhTransform->joint[jID].pos3D[2] = (float) pos3DCenterFloat[2];               );
         }
           else
         {
         //-------------------------------------------------------------------------------------------
         //Or we directly render using the matrices in our simplerenderer
         //-------------------------------------------------------------------------------------------
           fprintf(stderr, "DIRECT RENDER ");
           simpleRendererRenderUsingPrecalculatedMatrices(
                                                          renderer,
                                                          pos3DFloat,
                                                          &position2D[0],
                                                          &position2D[1],
                                                          &position2D[2]
                                                         );
          }
         //-------------------------------------------------------------------------------------------
           if (position2D[2]>=0.0)
           {
              bvhTransform->joint[jID].pos2D[0] = (float) position2D[0];
              bvhTransform->joint[jID].pos2D[1] = (float) position2D[1];
              bvhTransform->joint[jID].pos2DCalculated=1;
              bvhTransform->joint[jID].isBehindCamera =0;
           } else
           {
              bvhTransform->joint[jID].pos2D[0] = 0.0;
              bvhTransform->joint[jID].pos2D[1] = 0.0;
              bvhTransform->joint[jID].pos2DCalculated=1;
              bvhTransform->joint[jID].isBehindCamera=1;
           }
        }// bvh_shouldJointBeTransformedGivenOurOptimizations
      } //Joint Loop , render all joint points..

    //------------------------------------------------------------------------------------------
    if (occlusions)
     {
      //Also project our Torso coordinates..
       bvh_populateRectangle2DFromProjections(mc,bvhTransform,&bvhTransform->torso);
       bvh_projectTo2DHandleOcclusions(mc,bvhTransform);
     //------------------------------------------------------------------------------------------
     }//End of occlusion code

 return (mc->jointHierarchySize>0);
}
