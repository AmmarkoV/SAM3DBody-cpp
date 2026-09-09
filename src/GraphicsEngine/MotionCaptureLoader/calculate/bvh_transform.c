#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bvh_transform.h"

#include "../mathLibrary.h"

//Also find Center of Joint
//We can skip the matrix multiplication by just grabbing the last column..
#define FAST_OFFSET_TRANSLATION 1

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

static inline float bvh_max(float a,float b)
{
  if (a>b) {return a;}
  return b;
}

static inline float bvh_min(float a,float b)
{
  if (a<b) {return a;}
  return b;
}

struct Vector3D
{
    float x;
    float y;
    float z;
};

struct Point3D
{
    float x;
    float y;
    float z;
};

struct Triangle3D
{
    struct Point3D p1;
    struct Point3D p2;
    struct Point3D p3;
};

/*
import sys
import os
import numpy as np

def point_triangle_distance(point, triangle):
    # Define the triangle vertices
    A = np.array(triangle[0])
    B = np.array(triangle[1])
    C = np.array(triangle[2])

    # Define the triangle normal
    normal = np.cross(B - A, C - A)
    normal = normal / np.linalg.norm(normal)

    # Define the point vector
    P = np.array(point)

    # Calculate the distance between the point and the triangle
    distance = np.dot(normal, (P - A))

    # Calculate the side of the triangle
    side = np.dot(normal, np.cross(B - A, P - A))

    # Return the signed distance depending on the side
    if side > 0:
        return distance
    else:
        return -distance




#----------------------------------------------
point = [14.85,32.82,-120.67]
P0 = [8.02,38.77,-158.73]
P1 = [-0.82,-2.51,-190.33]
P2 = [16.70,-1.84,-191.22]
n = [0.06,-0.62,0.79]

#----------------------------------------------
triangle = [P0,P1,P2]
print(point_triangle_distance(point,triangle))
sys.exit(0)

*/

struct Vector3D cross_product(struct Vector3D A, struct Vector3D B)
{
    struct Vector3D cross_product;
    cross_product.x = A.y * B.z - A.z * B.y;
    cross_product.y = A.z * B.x - A.x * B.z;
    cross_product.z = A.x * B.y - A.y * B.x;
    return cross_product;
}

float dot_product(struct Vector3D A, struct Vector3D B)
{
    return A.x * B.x + A.y * B.y + A.z * B.z;
}

float bvh_DistanceOfJointFromTorsoPlane(struct BVH_MotionCapture * mc,
                                        struct BVH_Transform * bvhTransform,
                                        BVHJointID jID)
{

/*
High quality ASCII art TM to understand the following..


   *   * P        *
   |     \       / \  n
d  |       \      |
i  |         \    |
s  |           \  |
t  *             Po
               /    \
         U   /        \ V
           /            \
         /                \ P2
        * P1

*/

 float dist = 0.0;
 if ( bvhTransform->torsoTriangle.exists )
  {
    struct Point3D point = { bvhTransform->joint[jID].pos3D[0], bvhTransform->joint[jID].pos3D[1], bvhTransform->joint[jID].pos3D[2] };
    //-------------------------------------------------------------------------------------------------------------------------------------------------------------
    struct Point3D p0 = { bvhTransform->torsoTriangle.triangle3D.x1, bvhTransform->torsoTriangle.triangle3D.y1, bvhTransform->torsoTriangle.triangle3D.z1 };
    struct Point3D p1 = { bvhTransform->torsoTriangle.triangle3D.x2, bvhTransform->torsoTriangle.triangle3D.y2, bvhTransform->torsoTriangle.triangle3D.z2 };
    struct Point3D p2 = { bvhTransform->torsoTriangle.triangle3D.x3, bvhTransform->torsoTriangle.triangle3D.y3, bvhTransform->torsoTriangle.triangle3D.z3 };
    //-------------------------------------------------------------------------------------------------------------------------------------------------------------
    struct Triangle3D triangle = { p0 , p1 ,p2 };
    //-------------------------------------------------------------------------------------------------------------------------------------------------------------
    struct Vector3D p = { point.x - triangle.p1.x, point.y - triangle.p1.y, point.z - triangle.p1.z };
    struct Vector3D U = { triangle.p2.x - triangle.p1.x, triangle.p2.y - triangle.p1.y, triangle.p2.z - triangle.p1.z };
    struct Vector3D V = { triangle.p3.x - triangle.p1.x, triangle.p3.y - triangle.p1.y, triangle.p3.z - triangle.p1.z };

    //First compute the normal vector of the triangle using the vectors pointing from triangle.p1 to triangle.p2 and triangle.p1 to triangle.p3.
    struct Vector3D n = {(U.y*V.z)-(U.z*V.y) , (U.z*V.x)-(U.x*V.z) , (U.x*V.y)-(U.y*V.x) };

    //Then computes the distance between the point and the plane of the triangle using the dot product.
    //Normalize
     dist = sqrt( (n.x*n.x) + (n.y*n.y) + (n.z*n.z) ); // normalized
     n.x /= dist;
     n.y /= dist;
     n.z /= dist;

     //Absolute Distance! fabs
     // Calculate the distance between the point and the triangle
      dist = dot_product(n,p);// np.dot(normal, (P - A))
      //dist = fabs ( (point.x - p0.x)*n.x + (point.y-p0.y)*n.y + (point.z-p0.z)*n.z ); // your absolute perpendicular distance

      //Explicit side check : side = np.dot(normal, np.cross(B - A, P - A))
      struct Vector3D cross = cross_product(U,p);
      float side = dot_product(n,cross);

      if (side<0)
          { dist = -1* dist; } //(n.x * (point.x - p0.x)) + (n.y * (point.y - p0.y)) + (n.z * (point.z - p0.z));

  //if (dist<0.0)
  if (0) //<- DEBUG
  {   //Python debug code..
      fprintf(stderr,"#for joint %s we have : \n",mc->jointHierarchy[jID].jointName);
      fprintf(stderr,"point = [%0.2f,%0.2f,%0.2f] \n",point.x,point.y,point.z);
      //------------------------------------------------------------------------------
      fprintf(stderr,"P0 = [%0.2f,%0.2f,%0.2f] \n",p0.x,p0.y,p0.z);
      fprintf(stderr,"P1 = [%0.2f,%0.2f,%0.2f] \n",p1.x,p1.y,p1.z);
      fprintf(stderr,"P2 = [%0.2f,%0.2f,%0.2f] \n",p2.x,p2.y,p2.z);
      //------------------------------------------------------------------------------
      fprintf(stderr,"n = [%0.2f,%0.2f,%0.2f] \n",n.x,n.y,n.z);
      fprintf(stderr,"dist = %0.2f \n",dist);
  }

  }

  return dist;
}


int bvh_populateTorso3DFromTransform(
                                      struct BVH_MotionCapture * mc ,
                                      struct BVH_Transform * bvhTransform
                                    )
{
 //Cleanup torso rectangle
 bvhTransform->torso.exists=0;
 bvhTransform->torso.point1Exists=0;
 bvhTransform->torso.point2Exists=0;
 bvhTransform->torso.point3Exists=0;
 bvhTransform->torso.point4Exists=0;
 bvhTransform->torso.rectangle2D.calculated=0;

 //Cleanup torso triangle
 bvhTransform->torsoTriangle.exists=0;
 bvhTransform->torsoTriangle.point1Exists=0;
 bvhTransform->torsoTriangle.point2Exists=0;
 bvhTransform->torsoTriangle.point3Exists=0;

 unsigned int jID=0;
 //Second test occlusions with torso..!
       //-------------------------------------------------------------
       int found=0;
       if ( bvh_getJointIDFromJointName(mc,"lshoulder",&jID) ) { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"lShldr",&jID) )    { found=1; }

       if (
            (found) &&
            (
              ( bvhTransform->joint[jID].pos3D[0]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[1]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[2]!=0.0 )
            )
          )
       {
           bvhTransform->torso.point1Exists=1;
           bvhTransform->torso.rectangle3D.x1=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y1=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z1=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[0]=jID;
       }
       //---
       found=0;
       if ( bvh_getJointIDFromJointName(mc,"rshoulder",&jID) ) { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"rShldr",&jID) )    { found=1; }

       if (
            (found) &&
            (
              ( bvhTransform->joint[jID].pos3D[0]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[1]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[2]!=0.0 )
            )
          )
       {
           bvhTransform->torso.point2Exists=1;
           bvhTransform->torso.rectangle3D.x2=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y2=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z2=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[1]=jID;
       }
       //---
       found=0;
       if ( bvh_getJointIDFromJointName(mc,"rhip",&jID) )      { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"rThigh",&jID) )    { found=1; }

       if (
            (found) &&
            (
              ( bvhTransform->joint[jID].pos3D[0]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[1]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[2]!=0.0 )
            )
          )
       {
           //--------------------------------------------------------------------------
           bvhTransform->torso.point3Exists=1;
           bvhTransform->torso.rectangle3D.x3=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y3=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z3=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[2]=jID;
           //--------------------------------------------------------------------------
           bvhTransform->torsoTriangle.point2Exists=1;
           bvhTransform->torsoTriangle.triangle3D.x2=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torsoTriangle.triangle3D.y2=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torsoTriangle.triangle3D.z2=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torsoTriangle.jID[1]=jID;
           //--------------------------------------------------------------------------
       }
       //---
       found=0;
       if ( bvh_getJointIDFromJointName(mc,"lhip",&jID) )      { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"lThigh",&jID) )    { found=1; }

       if (
            (found) &&
            (
              ( bvhTransform->joint[jID].pos3D[0]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[1]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[2]!=0.0 )
            )
          )
       {
           //--------------------------------------------------------------------------
           bvhTransform->torso.point4Exists=1;
           bvhTransform->torso.rectangle3D.x4=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torso.rectangle3D.y4=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torso.rectangle3D.z4=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torso.jID[3]=jID;
           //--------------------------------------------------------------------------
           bvhTransform->torsoTriangle.point3Exists=1;
           bvhTransform->torsoTriangle.triangle3D.x3=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torsoTriangle.triangle3D.y3=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torsoTriangle.triangle3D.z3=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torsoTriangle.jID[2]=jID;
           //--------------------------------------------------------------------------
       }
       //---

       if (
            (bvhTransform->torso.point1Exists) &&
            (bvhTransform->torso.point2Exists) &&
            (bvhTransform->torso.point3Exists) &&
            (bvhTransform->torso.point4Exists)
          )
         {
            bvhTransform->torso.exists=1;
            bvhTransform->torso.averageDepth = ( bvhTransform->torso.rectangle3D.z1 +
                                                 bvhTransform->torso.rectangle3D.z2 +
                                                 bvhTransform->torso.rectangle3D.z3 +
                                                 bvhTransform->torso.rectangle3D.z4 ) / 4;
         }


       found=0;
       if ( bvh_getJointIDFromJointName(mc,"neck",&jID) )      { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"neck1",&jID) )     { found=1; } else
       if ( bvh_getJointIDFromJointName(mc,"neck01",&jID) )    { found=1; }

       if (
            (found) &&
            (
              ( bvhTransform->joint[jID].pos3D[0]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[1]!=0.0 ) ||
              ( bvhTransform->joint[jID].pos3D[2]!=0.0 )
            )
          )
       {
           //--------------------------------------------------------------------------
           bvhTransform->torsoTriangle.point1Exists=1;
           bvhTransform->torsoTriangle.triangle3D.x1=bvhTransform->joint[jID].pos3D[0];
           bvhTransform->torsoTriangle.triangle3D.y1=bvhTransform->joint[jID].pos3D[1];
           bvhTransform->torsoTriangle.triangle3D.z1=bvhTransform->joint[jID].pos3D[2];
           bvhTransform->torsoTriangle.jID[0]=jID;
           //--------------------------------------------------------------------------
       }
       //-------------------------------------------------------------

       if (
            (bvhTransform->torsoTriangle.point1Exists) &&
            (bvhTransform->torsoTriangle.point2Exists) &&
            (bvhTransform->torsoTriangle.point3Exists)
          )
         {
            bvhTransform->torsoTriangle.exists=1;
            bvhTransform->torsoTriangle.averageDepth = ( bvhTransform->torsoTriangle.triangle3D.z1 +
                                                         bvhTransform->torsoTriangle.triangle3D.z2 +
                                                         bvhTransform->torsoTriangle.triangle3D.z3 ) / 3;
         }
  //fprintf(stderr,"%u %u %u %u\n",bvhTransform->torso.point1Exists,bvhTransform->torso.point2Exists,bvhTransform->torso.point3Exists,bvhTransform->torso.point4Exists);
  return 0;
}




int bvh_populateRectangle2DFromProjections(
                                           struct BVH_MotionCapture * mc ,
                                           struct BVH_Transform * bvhTransform,
                                           struct rectangleArea * area
                                          )
{
  if (area->exists)
  {
    unsigned int jID=0;
    unsigned int existing2DPoints=0;

    //-----------------------------------------------------------
    jID=area->jID[0];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x1=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y1=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }
    //-----------------------------------------------------------
    jID=area->jID[1];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x2=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y2=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }

    //-----------------------------------------------------------
    jID=area->jID[2];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x3=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y3=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }

    //-----------------------------------------------------------
    jID=area->jID[3];
    if (bvhTransform->joint[jID].pos2DCalculated)
    {
        area->rectangle2D.x4=bvhTransform->joint[jID].pos2D[0];
        area->rectangle2D.y4=bvhTransform->joint[jID].pos2D[1];
        ++existing2DPoints;
    }

    if ( existing2DPoints == 4 )
    {
      area->rectangle2D.calculated=1;

      float minimumX=bvh_min(area->rectangle2D.x1,bvh_min(area->rectangle2D.x2,bvh_min(area->rectangle2D.x3,area->rectangle2D.x4)));
      float minimumY=bvh_min(area->rectangle2D.y1,bvh_min(area->rectangle2D.y2,bvh_min(area->rectangle2D.y3,area->rectangle2D.y4)));
      float maximumX=bvh_max(area->rectangle2D.x1,bvh_max(area->rectangle2D.x2,bvh_max(area->rectangle2D.x3,area->rectangle2D.x4)));
      float maximumY=bvh_max(area->rectangle2D.y1,bvh_max(area->rectangle2D.y2,bvh_max(area->rectangle2D.y3,area->rectangle2D.y4)));
      area->rectangle2D.x=minimumX;
      area->rectangle2D.y=minimumY;
      area->rectangle2D.width=maximumX-minimumX;
      area->rectangle2D.height=maximumY-minimumY;
      return 1;
    } else
    {
        fprintf(stderr,"Only found %u/4 of the joints needed to get a rectangle \n",existing2DPoints);
    }
  } else
  {
    //Less spam..
   //fprintf(stderr,"bvh_populateRectangle2DFromProjections: Area does not exist..\n");
  }

 return 0;
}




unsigned char bvh_shouldJointBeTransformedGivenOurOptimizations(const struct BVH_Transform * bvhTransform,const BVHJointID jID)
{
  //This call is called millions of times during IK so it is really important to be fast

  //                                                                                           || <- Using + instead of ||
  //                                                                                           \/
  return (((bvhTransform->useOptimizations) && (!bvhTransform->skipCalculationsForJoint[jID])) + (!bvhTransform->useOptimizations) );

  //The old implementation is retained here for historical reasons :P
  //unsigned char res;
  //Normaly we should check if the bvhTransform structure is null but given that this call is called millions of times only if you have a transform
  //we skip the check
  //if (bvhTransform!=0)
  //  {
  //      switch(bvhTransform->useOptimizations)
  //      {
  //        case 1:
  //            res = !bvhTransform->skipCalculationsForJoint[jID];
  //        break;
  //        default :
  //            res = 1;
  //        break;
  //     };

        //if (bvhTransform->useOptimizations) // Deactivating this improves IK by 0.3%
        //{
         //If we are using optimizations and this joint is not skipped then transform this joint
         //Normally we should check for index errors but in an effort to speed up the function to the maximum extent the check is skipped
         //if (jID<bvhTransform->numberOfJointsToTransform)  //<- check can be disabled for speedup of + 0.5%
              //{
                 //res = !bvhTransform->skipCalculationsForJoint[jID];
              //}
          //return 0;
        //}
  //  }
  //return res;
}



void bvh_printBVHTransform(const char * label,struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform)
{
   if (bvhMotion==0)
   {
       fprintf(stderr,"bvh_printBVHTransform, no bvhMotion..!\n");
       return;
   }
   if (bvhTransform==0)
   {
       fprintf(stderr,"bvh_printBVHTransform, no bvhTransform..!\n");
       return;
   }

   fprintf(stderr,"bvh_printBVHTransform for %s\n",label);
   fprintf(stderr,"jointHierarchySize=%u\n",bvhMotion->jointHierarchySize);
   fprintf(stderr,"useOptimizations=%u\n",(unsigned int) bvhTransform->useOptimizations);

   fprintf(stderr,"skipCalculationsForJoint:\n");
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
      if (bvhTransform->skipCalculationsForJoint[jID])
      {
         fprintf(stderr,"skipCalculationsForJoint[%u]=%u\n",jID,(unsigned int) bvhTransform->skipCalculationsForJoint[jID]);
      }
   }
   fprintf(stderr,"\n");


   fprintf(stderr,"jointIDTransformHashPopulated=%u\n",bvhTransform->jointIDTransformHashPopulated);
   fprintf(stderr,"lengthOfListOfJointIDsToTransform=%u\n",bvhTransform->lengthOfListOfJointIDsToTransform);

   fprintf(stderr,"listOfJointIDsToTransform:\n");
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
      if (bvhTransform->listOfJointIDsToTransform[jID])
      {
         fprintf(stderr,"listOfJointIDsToTransform[%u]=%u\n",jID,bvhTransform->listOfJointIDsToTransform[jID]);
      }
   }
   fprintf(stderr,"\n");

   fprintf(stderr,"jointsOccludedIn2DProjection=%u\n",bvhTransform->jointsOccludedIn2DProjection);
   fprintf(stderr,"centerPosition[3]={%0.2f,%0.2f,%0.2f}\n",bvhTransform->centerPosition[0],bvhTransform->centerPosition[1],bvhTransform->centerPosition[2]);

    for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
      //if (bvhTransform->listOfJointIDsToTransform[jID])
      {
         fprintf(stderr,"joint[%u]={\n",jID);
         fprintf(stderr,"            name=%s\n",bvhMotion->jointHierarchy[jID].jointName);
         fprintf(stderr,"            bvhTransform->joint[%u].pos2DCalculated=%u\n",jID,(unsigned int) bvhTransform->joint[jID].pos2DCalculated);
         fprintf(stderr,"            bvhTransform->joint[%u].isBehindCamera=%u\n",jID,(unsigned int) bvhTransform->joint[jID].isBehindCamera);
         fprintf(stderr,"            bvhTransform->joint[%u].isOccluded=%u\n",jID,(unsigned int) bvhTransform->joint[jID].isOccluded);
         fprintf(stderr,"            bvhTransform->joint[%u].isChainTransformationComputed=%u\n",jID,(unsigned int) bvhTransform->joint[jID].isChainTransformationComputed);

         fprintf(stderr,"\n            bvhTransform->joint[%u].pos2D={%0.2f,%0.2f}\n",jID,bvhTransform->joint[jID].pos2D[0],bvhTransform->joint[jID].pos2D[1]);
         fprintf(stderr,"\n            bvhTransform->joint[%u].pos3D={%0.2f,%0.2f,%0.2f,%0.2f}\n",jID,bvhTransform->joint[jID].pos3D[0],bvhTransform->joint[jID].pos3D[1],bvhTransform->joint[jID].pos3D[2],bvhTransform->joint[jID].pos3D[3]);

         print4x4FMatrix("localToWorldTransformation",bvhTransform->joint[jID].localToWorldTransformation.m,1);
         fprintf(stderr,"\n");
         print4x4FMatrix("chainTransformation",bvhTransform->joint[jID].chainTransformation.m,1);
         fprintf(stderr,"\n");
         print4x4FMatrix("dynamicTranslation",bvhTransform->joint[jID].dynamicTranslation.m,1);
         fprintf(stderr,"\n");
         print4x4FMatrix("dynamicRotation",bvhTransform->joint[jID].dynamicRotation.m,1);
         fprintf(stderr,"\n");

         fprintf(stderr,"}\n");
      }
   }
}



void bvh_printNotSkippedJoints(struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform)
{
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     if (!bvhTransform->skipCalculationsForJoint[jID])
     {
         fprintf(stderr,"Joint %u ( %s ) is selected \n" ,jID , bvhMotion->jointHierarchy[jID].jointName);
     }
   }
}



void bvh_HashUsefulJoints(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform)
{
   #if USE_TRANSFORM_HASHING
   /*
   //This is a little stupid.. if we are not using optimizations then we go through
   //all elements..
   if (!bvhTransform->useOptimizations)
   {
      unsigned int hashedElementsCounter=0;
      for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
       {
         bvhTransform->listOfJointIDsToTransform[hashedElementsCounter]=jID;
         ++hashedElementsCounter;
       }

     bvhTransform->lengthOfListOfJointIDsToTransform=hashedElementsCounter; //Start from the .. start..
     bvhTransform->jointIDTransformHashPopulated=1; //(hashedElementsCounter>0); //Mark the hash as populated
     return ;
   }
   */

   //Since we do several million accesses an additional optimization is to keep a list of interesting joints
   //and only access them
   //-----------------------------------------------------------------------------------------------
   unsigned int hashedElementsCounter=0;
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
       if ( (!bvhTransform->useOptimizations) || (!bvhTransform->skipCalculationsForJoint[jID]) )
       {
         bvhTransform->listOfJointIDsToTransform[hashedElementsCounter]=jID;
         ++hashedElementsCounter;
       }
   }
   bvhTransform->lengthOfListOfJointIDsToTransform=hashedElementsCounter; //Start from the .. start..
   bvhTransform->jointIDTransformHashPopulated=1; //(hashedElementsCounter>0); //Mark the hash as populated
   //-----------------------------------------------------------------------------------------------
   #endif
   return;
}



int bvh_markAllJointsAsUsefullInTransform(
                                          struct BVH_MotionCapture * bvhMotion ,
                                          struct BVH_Transform * bvhTransform
                                         )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  bvhTransform->useOptimizations=0; //When all joints are useful in a nutshell, we don't use optimizations..

   //But we also set the skip flag for each joint to zero in case the useOptimization flag is toggled somewhere else
   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     bvhTransform->skipCalculationsForJoint[jID]=0;
   }

  #if USE_TRANSFORM_HASHING
   //Marking all joints as useless is the same as setting the length of the joint
   //list to zero
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif

  return 1;
}


int bvh_markAllJointsAsUselessInTransform(
                                          struct BVH_MotionCapture * bvhMotion,
                                          struct BVH_Transform * bvhTransform
                                         )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (bvhMotion->jointHierarchySize>bvhTransform->numberOfJointsSpaceAllocated)
       {
         fprintf(stderr,"bvh_markAllJointsAsUselessInTransform error %u/%u..\n ",bvhMotion->jointHierarchySize,bvhTransform->numberOfJointsSpaceAllocated);
         return 0;
       }
  bvhTransform->useOptimizations=1;

   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     bvhTransform->skipCalculationsForJoint[jID]=1;
   }

  #if USE_TRANSFORM_HASHING
   //Marking all joints as useless is the same as setting the length of the joint
   //list to zero
   bvhTransform->lengthOfListOfJointIDsToTransform=0; //No joints are interesting
  #endif

  return 1;
}


int bvh_markJointAndParentsAsUsefulInTransform(
                                                struct BVH_MotionCapture * bvhMotion,
                                                struct BVH_Transform * bvhTransform,
                                                BVHJointID jID
                                              )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (jID>=bvhMotion->jointHierarchySize) { return 0; }
  bvhTransform->useOptimizations=1;

  //We want to make sure all parent joints until root ( jID->0 ) are set to not skip calculations..
  while (jID!=0)
      {
           if (jID<bvhTransform->numberOfJointsSpaceAllocated)
           {
            bvhTransform->skipCalculationsForJoint[jID]=0;
            jID = bvhMotion->jointHierarchy[jID].parentJoint;
           } else
           {
             fprintf(stderr,"bvh_markJointAndParentsAsUsefulInTransform: invalid jID encountered while traversing parents (%u/%u)\n",jID,bvhTransform->numberOfJointsSpaceAllocated);
             break;
           }
      }

  bvhTransform->skipCalculationsForJoint[0]=0;

  #if USE_TRANSFORM_HASHING
  //As an extra speed up we hash the interesting joints
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif

  return 1;
}


int bvh_markJointAndParentsAsUselessInTransform(
                                                struct BVH_MotionCapture * bvhMotion,
                                                struct BVH_Transform * bvhTransform,
                                                BVHJointID jID
                                              )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (jID>=bvhMotion->jointHierarchySize) { return 0; }
  bvhTransform->useOptimizations=1;

  while (jID!=0)
      {
           if (jID<bvhTransform->numberOfJointsSpaceAllocated)
           {
            bvhTransform->skipCalculationsForJoint[jID]=1;
            jID = bvhMotion->jointHierarchy[jID].parentJoint;
           } else
           {
             fprintf(stderr,"bvh_markJointAndParentsAsUselessInTransform: invalid jID encountered while traversing parents\n");
             break;
           }
      }

  bvhTransform->skipCalculationsForJoint[0]=1;

  #if USE_TRANSFORM_HASHING
   //As an extra speed up we hash the interesting joints
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif

  return 1;
}


int bvh_markJointAsUsefulAndParentsAsUselessInTransform(
                                                        struct BVH_MotionCapture * bvhMotion,
                                                        struct BVH_Transform * bvhTransform,
                                                        BVHJointID jID
                                                       )
{
  if (bvhMotion==0)    { return 0; }
  if (bvhTransform==0) { return 0; }
  if (jID>=bvhMotion->jointHierarchySize) { return 0; }

  bvhTransform->useOptimizations=1;
  bvh_markJointAndParentsAsUselessInTransform(bvhMotion,bvhTransform,jID);
  bvhTransform->skipCalculationsForJoint[jID]=0;

  #if USE_TRANSFORM_HASHING
   //As an extra speed up we hash the interesting joints
   bvh_HashUsefulJoints(bvhMotion,bvhTransform);
  #endif

  return 1;
}




static inline void bvh_prepareMatricesForTransform(
                                                   struct BVH_MotionCapture * bvhMotion,
                                                   float * motionBuffer,
                                                   struct BVH_Transform * bvhTransform,
                                                   unsigned int jID
                                                  )
{
  //data is the buffer where we will retrieve the values
  float data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER]={0};
  //----------------------------------------------------

  //To Setup the dynamic transformation we must first get values from our bvhMotion structure
  if (bvh_retrieveDataFromMotionBuffer(bvhMotion,jID,motionBuffer,data,sizeof(data)))
      {
       create4x4FTranslationMatrix(
                                    &bvhTransform->joint[jID].dynamicTranslation,
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X],
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y],
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z]
                                  );


      if ( (bvhMotion->jointHierarchy[jID].channelRotationOrder!=BVH_ROTATION_ORDER_NONE) )
       {
          if(bvhMotion->jointHierarchy[jID].hasRodriguesRotation)
          {
            //fprintf(stderr,"Rodrigues Transformation %f %f %f\n",-1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],-1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],-1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]);
            create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                              &bvhTransform->joint[jID].dynamicRotation,
                                                             -1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],
                                                             -1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],
                                                             -1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z],
                                                             BVH_ROTATION_ORDER_RODRIGUES
                                                            );
          } else
          if(bvhMotion->jointHierarchy[jID].hasQuaternionRotation)
          {
            //BVH Quaternion..
            float quaternion[4]={
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_W],
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],
                                    data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]
                                };

            //Make sure quaternion is normalized otherwise conversion will fail on next step..
            normalizeQuaternions(&quaternion[1],&quaternion[2],&quaternion[3],&quaternion[0]);

            quaternion2Matrix4x4(
                                    bvhTransform->joint[jID].dynamicRotation.m,
                                    quaternion,
                                    qWqXqYqZ
                                );
          } else
          { //Generic rotation case..
            unsigned int channelRotationOrder = (unsigned int) bvhMotion->jointHierarchy[jID].channelRotationOrder;
            create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                             &bvhTransform->joint[jID].dynamicRotation,
                                                             -1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],
                                                             -1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],
                                                             -1*data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z],
                                                             channelRotationOrder
                                                            );
          }
       } else
       {
         //No rotation order will get you an Identity rotation
         create4x4FIdentityMatrix(&bvhTransform->joint[jID].dynamicRotation);

         //This is normal in an end-site but we should
         //display a debug message in case this doesn't happen on an end site
         if (!bvhMotion->jointHierarchy[jID].isEndSite)
              {
                fprintf(stderr,"No channel rotation order for joint jID=%u jointName=%s, using identity matrix\n",jID,bvhMotion->jointHierarchy[jID].jointName);
              }
       }
     } else
     {
      fprintf(stderr,"Error extracting dynamic transformation for jID=%u and a motionBuffer\n",jID);
      create4x4FIdentityMatrix(&bvhTransform->joint[jID].dynamicTranslation);
      create4x4FIdentityMatrix(&bvhTransform->joint[jID].dynamicRotation);
     }
  return;
}


static inline void bvh_performActualTransform(
                                              struct BVH_MotionCapture * bvhMotion,
                                              float * motionBuffer,
                                              struct BVH_Transform * bvhTransform,
                                              unsigned int jID
                                             )
{
  if (!bvhMotion->jointHierarchy[jID].isRoot) //(bvh_jointHasParent(bvhMotion,jID))
      {
        //If joint is not Root joint
        unsigned int parentID = bvhMotion->jointHierarchy[jID].parentJoint;

        if (bvhTransform->joint[parentID].isChainTransformationComputed)
        {
         if (!bvhMotion->jointHierarchy[jID].hasPositionalChannels)
         {
          multiplyTwo4x4FMatricesS(
                                  //Output AxB
                                  &bvhTransform->joint[jID].localToWorldTransformation,
                                  //Parent Output A
                                  &bvhTransform->joint[parentID].chainTransformation,
                                  //This Transform B
                                  &bvhMotion->jointHierarchy[jID].staticTransformation
                                 );
         } else
         {
          //Special case where joint has positional channels..
          //We will ignore our static Transform and just use the positional channels encountered
          multiplyTwo4x4FMatricesS(
                                  //Output AxB
                                  &bvhTransform->joint[jID].localToWorldTransformation,
                                  //Parent Output A
                                  &bvhTransform->joint[parentID].chainTransformation,
                                  //This Transform B
                                  &bvhTransform->joint[jID].dynamicTranslation
                                 );
         }
        } else
        {
         //This is needed because we access the chain transform of our parent so at some point this will get used..
         if (!bvhMotion->jointHierarchy[jID].hasPositionalChannels)
          { copy4x4FMatrix(bvhTransform->joint[jID].localToWorldTransformation.m , bvhMotion->jointHierarchy[jID].staticTransformation.m); } else
          { copy4x4FMatrix(bvhTransform->joint[jID].localToWorldTransformation.m , bvhTransform->joint[jID].dynamicTranslation.m);         }
        }
      } // not bvhMotion->jointHierarchy[jID].isRoot
       else
      {
       //If we are the root node there is no parent so we skip the multiplication with the "Identity" chainTransformation..
       //If there is no parent we will only set our position and copy to the final transform
       //Skip the matrix multiplication..
       create4x4FTranslationMatrix(
                                     &bvhTransform->joint[jID].localToWorldTransformation,
                                     bvhMotion->jointHierarchy[jID].staticTransformation.m[3]  + bvhTransform->joint[jID].dynamicTranslation.m[3],
                                     bvhMotion->jointHierarchy[jID].staticTransformation.m[7]  + bvhTransform->joint[jID].dynamicTranslation.m[7],
                                     bvhMotion->jointHierarchy[jID].staticTransformation.m[11] + bvhTransform->joint[jID].dynamicTranslation.m[11]
                                  );
      }

    //Calculate chain transformation for this jID
    //----------------------------------------------------------------------------
    multiplyTwo4x4FMatricesS(
                             &bvhTransform->joint[jID].chainTransformation,         //Output AxB
                             &bvhTransform->joint[jID].localToWorldTransformation,  //A
                             &bvhTransform->joint[jID].dynamicRotation              //B
                            );
    bvhTransform->joint[jID].isChainTransformationComputed=1;


   //Also do 3D position output calculation..
   //------------------------------------------------------------------------------------
   bvhTransform->joint[jID].pos3D[0]=bvhTransform->joint[jID].chainTransformation.m[3];
   bvhTransform->joint[jID].pos3D[1]=bvhTransform->joint[jID].chainTransformation.m[7];
   bvhTransform->joint[jID].pos3D[2]=bvhTransform->joint[jID].chainTransformation.m[11];
   bvhTransform->joint[jID].pos3D[3]=bvhTransform->joint[jID].chainTransformation.m[15];
   //------------------------------------------------------------------------------------
   if (bvhTransform->joint[jID].pos3D[3]!=1.0)
      { normalize3DPointFVector(bvhTransform->joint[jID].pos3D); }
   //------------------------------------------------------------------------------------
   return;
}


int bvh_allocateTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform)
{
  #if DYNAMIC_TRANSFORM_ALLOCATIONS
  if ( (bvhMotion!=0) &&  (bvhTransform!=0))
  {
    //Two cases when we want to allocate a BVH Transform.
    if (
         (bvhTransform->numberOfJointsSpaceAllocated>=bvhMotion->MAX_jointHierarchySize) &&
         (bvhTransform->skipCalculationsForJoint!=0) &&
         (bvhTransform->joint!=0) &&
         (bvhTransform->listOfJointIDsToTransform!=0)
       )
       {
         //First case, smart and fast path is that if we already have enough space allocated we change nothing but the number of joints
         //that we want to transform, this way there is no reallocations on multiple sequencial transforms..
         bvhTransform->numberOfJointsToTransform = bvhMotion->jointHierarchySize;
       } else
       {
         //Second case, force reallocations and cleanup of data..
         bvhTransform->numberOfJointsSpaceAllocated = bvhMotion->MAX_jointHierarchySize;
         bvhTransform->numberOfJointsToTransform = bvhMotion->jointHierarchySize;

         if (bvhTransform->skipCalculationsForJoint!=0)  { free(bvhTransform->skipCalculationsForJoint);  bvhTransform->skipCalculationsForJoint=0; }
         if (bvhTransform->joint!=0)                     { free(bvhTransform->joint);                     bvhTransform->joint=0; }
         if (bvhTransform->listOfJointIDsToTransform!=0) { free(bvhTransform->listOfJointIDsToTransform); bvhTransform->listOfJointIDsToTransform=0; }

         bvhTransform->skipCalculationsForJoint  = (unsigned char *)               malloc(sizeof(unsigned char)               * bvhTransform->numberOfJointsSpaceAllocated);
         bvhTransform->joint                     = (struct BVH_TransformedJoint *) malloc(sizeof(struct BVH_TransformedJoint) * bvhTransform->numberOfJointsSpaceAllocated);

         ///https://software.intel.com/content/www/us/en/develop/articles/coding-for-performance-data-alignment-and-structures.html
         // This should probably use _mm_malloc! arr1=(struct s2 *)_mm_malloc(sizeof(struct s2)*my_size,32);
         bvhTransform->listOfJointIDsToTransform = (BVHJointID *)                  malloc(sizeof(BVHJointID)                  * bvhTransform->numberOfJointsSpaceAllocated);

         if (bvhTransform->skipCalculationsForJoint!=0)
              { memset(bvhTransform->skipCalculationsForJoint, 0,sizeof(unsigned char) * bvhTransform->numberOfJointsSpaceAllocated); }

         if (bvhTransform->joint!=0)
              { memset(bvhTransform->joint,                    0,sizeof(struct BVH_TransformedJoint) * bvhTransform->numberOfJointsSpaceAllocated); }

         if (bvhTransform->listOfJointIDsToTransform!=0)
              { memset(bvhTransform->listOfJointIDsToTransform,0,sizeof(BVHJointID) * bvhTransform->numberOfJointsSpaceAllocated); }
       }

    bvhTransform->transformStructInitialized = ( (bvhTransform->skipCalculationsForJoint!=0) && (bvhTransform->joint!=0) && (bvhTransform->listOfJointIDsToTransform!=0) );

    return bvhTransform->transformStructInitialized;
  }
  #else
      bvhTransform->numberOfJointsSpaceAllocated = MAX_BVH_TRANSFORM_SIZE;
      bvhTransform->numberOfJointsToTransform    = bvhMotion->jointHierarchySize;
      bvhTransform->transformStructInitialized   = 1;
   return 1;
  #endif
}



int bvh_freeTransform(struct BVH_Transform * bvhTransform)
{
    if(bvhTransform!=0)
    {
     #if DYNAMIC_TRANSFORM_ALLOCATIONS
      //fprintf(stderr,"bvh_freeTransform : ");
      if (bvhTransform->skipCalculationsForJoint!=0)  { free(bvhTransform->skipCalculationsForJoint);  bvhTransform->skipCalculationsForJoint=0; }
      if (bvhTransform->joint!=0)                     { free(bvhTransform->joint);                     bvhTransform->joint=0;                    }
      if (bvhTransform->listOfJointIDsToTransform!=0) { free(bvhTransform->listOfJointIDsToTransform); bvhTransform->listOfJointIDsToTransform=0;}
    #endif

     bvhTransform->transformStructInitialized=0;
     //fprintf(stderr,"survived..\n");
    }
    return 1;
}


int bvh_loadTransformForMotionBufferFollowingAListOfJointIDs(
                                                             struct BVH_MotionCapture * bvhMotion,
                                                             float * motionBuffer,
                                                             struct BVH_Transform * bvhTransform,
                                                             unsigned int populateTorso,
                                                             BVHJointID * listOfJointIDsToTransform,
                                                             unsigned int lengthOfJointIDList
                                                            )
{
  int res = 0;
  //Only do transforms on allocated context
  if ( (bvhMotion!=0) && (motionBuffer!=0) && (bvhTransform!=0))
  {
    //Make sure there is enough memory allocated..
    if (bvh_allocateTransform(bvhMotion,bvhTransform))
    {

      //First of all we need to clean the BVH_Transform structure
      bvhTransform->jointsOccludedIn2DProjection=0;


      //First of all we need to populate all local dynamic transformation of our chain
      //This step only has to do with our Motion Buffer and doesn't perform the final transformations
      //----------------------------------------------------------------------------------------
       for (unsigned int hID=0; hID<lengthOfJointIDList; hID++)
       {
        unsigned int jID=listOfJointIDsToTransform[hID];
        if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
        {
          bvh_prepareMatricesForTransform(bvhMotion,motionBuffer,bvhTransform,jID);
        }
       }

      //We will now apply all dynamic transformations across the BVH chains
      //-----------------------------------------------------------------------
      for (unsigned int hID=0; hID<lengthOfJointIDList; hID++)
      {
        unsigned int jID=listOfJointIDsToTransform[hID];
        if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
        {
          bvh_performActualTransform(
                                     bvhMotion,
                                     motionBuffer,
                                     bvhTransform,
                                     jID
                                    );
        }
       }
      //-----------------------------------------------------------------------------------
      bvhTransform->centerPosition[0]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[0];
      bvhTransform->centerPosition[1]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[1];
      bvhTransform->centerPosition[2]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[2];
      //-----------------------------------------------------------------------------------


      if (!populateTorso)
      {
        //Fast path out of here
        res = 1;
      } else
      {
       if (!bvh_populateTorso3DFromTransform(bvhMotion,bvhTransform))
         {
         //fprintf(stderr,"bvh_loadTransformForMotionBuffer: Could not populate torso information from 3D transform\n");
         }
        res = 1;
      }
  } else
  { fprintf(stderr,"Failed allocating memory for bvh trasnform :(\n"); }
  //--------------------------------------------------------------------
 }

 return res;
}




int bvh_loadTransformForMotionBuffer(
                                     struct BVH_MotionCapture * bvhMotion,
                                     float * motionBuffer,
                                     struct BVH_Transform * bvhTransform,
                                     unsigned int populateTorso
                                   )
{
  //Only do transforms on allocated context
  if ( (bvhMotion!=0) && (motionBuffer!=0) && (bvhTransform!=0))
  {
   //Make sure enough memory is allocated..
   if (!bvh_allocateTransform(bvhMotion,bvhTransform))
    {
      fprintf(stderr,RED "Failed allocating memory for bvh transform :(\n" NORMAL);
      return 0;
    }

  //First of all we need to clean the BVH_Transform structure
  bvhTransform->jointsOccludedIn2DProjection=0;


  //First of all we need to populate all local dynamic transformation of our chain
  //These only have to do with our Motion Buffer and don't involve any chain transformations
  //----------------------------------------------------------------------------------------

  //While doing this first step we accumulate the list of JointIDs that need a transform
  //This saves CPU time on the second part of this call as well as enables
  //economic subsequent calls on the IK code..
  bvhTransform->lengthOfListOfJointIDsToTransform=0;

  //Guard against worst case scenarios..
  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   if (bvhTransform->numberOfJointsSpaceAllocated<bvhMotion->jointHierarchySize)
   {
    fprintf(stderr,RED "Not enough space for joint IDs that need transform, aborting\n" NORMAL);
    return 0;
   }
  #else
   if (MAX_BVH_TRANSFORM_SIZE<bvhMotion->jointHierarchySize)
   {
    fprintf(stderr,RED "Not enough space for joint IDs that need transform, aborting\n" NORMAL);
    return 0;
   }
  #endif


   if (bvhTransform->useOptimizations)
   {
    //Two
    for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
     if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
     {
      //Since we are passing through make sure we avoid a second
      //"expensive" call to the bvh_shouldJointBeTransformedGivenOurOptimizations
      //this is also very important for the IK code..
      //------------------------------------------------------
       bvhTransform->listOfJointIDsToTransform[bvhTransform->lengthOfListOfJointIDsToTransform]=jID;
       ++bvhTransform->lengthOfListOfJointIDsToTransform;
      //------------------------------------------------------

      bvh_prepareMatricesForTransform(bvhMotion,motionBuffer,bvhTransform,jID);
     }
    }

   //We will now apply all dynamic transformations across the BVH chains
   //-----------------------------------------------------------------------
   //We used our buffer to cache only the joints that need processing during the first path
   //so we will only process those..
   for (unsigned int hID=0; hID<bvhTransform->lengthOfListOfJointIDsToTransform; hID++)
    {
      BVHJointID jID = bvhTransform->listOfJointIDsToTransform[hID];
      if (bvh_shouldJointBeTransformedGivenOurOptimizations(bvhTransform,jID))
      {
       bvh_performActualTransform(
                                  bvhMotion,
                                  motionBuffer,
                                  bvhTransform,
                                  jID
                                 );
      }
    }
   } //<- Do  transforms with optimisations enabled..!
     else
   {
      //If we don't want any optimizations just prepare all matrices
      for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
       {
         bvh_prepareMatricesForTransform(
                                          bvhMotion,
                                          motionBuffer,
                                          bvhTransform,
                                          jID
                                        );
       }

      //And then perform all transforms
      for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
       {
         bvh_performActualTransform(
                                    bvhMotion,
                                    motionBuffer,
                                    bvhTransform,
                                    jID
                                   );
       }
   }



  bvhTransform->centerPosition[0]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[0];
  bvhTransform->centerPosition[1]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[1];
  bvhTransform->centerPosition[2]=bvhTransform->joint[bvhMotion->rootJointID].pos3D[2];

  //Regardless of torso return 1!
  //return 1;


  if (!populateTorso)
  {
    //Fast path out of here
    return 1;
  } else
  {
   if (!bvh_populateTorso3DFromTransform(bvhMotion,bvhTransform))
     {
       //fprintf(stderr,"bvh_loadTransformForMotionBuffer: Could not populate torso information from 3D transform\n");
       //This is not a very important failure so w still consider it a success..
       //return 0;
     }
    return 1;
  }
 }

 return 0;
}



int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform,
                               unsigned int populateTorso
                             )
{
  int result = 0;
  if (bvh_allocateTransform(bvhMotion,bvhTransform))
    {
      struct MotionBuffer * frameMotionBuffer  = mallocNewMotionBuffer(bvhMotion);

      if (frameMotionBuffer!=0)
       {
         if  ( bvh_copyMotionFrameToMotionBuffer(bvhMotion,frameMotionBuffer,fID)  )
           {
            result = bvh_loadTransformForMotionBuffer(
                                                      bvhMotion ,
                                                      frameMotionBuffer->motion,
                                                      bvhTransform,
                                                      populateTorso
                                                    );
           } else
           {
             fprintf(stderr,RED "Could copy Motion frame to buffer for frame %u \n" NORMAL,fID);
           }

         freeMotionBuffer(&frameMotionBuffer);
       } else
       {
        fprintf(stderr,RED "Could not allocate memory for new motion buffer\n" NORMAL);
       }
    } else
    {
        fprintf(stderr,RED "Could not allocate memory for BVH Transform\n" NORMAL);
    }
  return result;
}


int bvh_removeTranslationFromTransform(
                                       struct BVH_MotionCapture * bvhMotion ,
                                       struct BVH_Transform * bvhTransform
                                      )
{
  fprintf(stderr,"bvh_removeTranslationFromTransform not correctly implemented");
  BVHJointID rootJID=0;

  if ( bvh_getRootJointID(bvhMotion,&rootJID) )
  {
   float rX = bvhTransform->joint[rootJID].pos3D[0];
   float rY = bvhTransform->joint[rootJID].pos3D[1];
   float rZ = bvhTransform->joint[rootJID].pos3D[2];

   float r2DX = bvhTransform->joint[rootJID].pos2D[0];
   float r2DY = bvhTransform->joint[rootJID].pos2D[1];

   for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
     bvhTransform->joint[jID].pos3D[0]=bvhTransform->joint[jID].pos3D[0]-rX;
     bvhTransform->joint[jID].pos3D[1]=bvhTransform->joint[jID].pos3D[1]-rY;
     bvhTransform->joint[jID].pos3D[2]=bvhTransform->joint[jID].pos3D[2]-rZ;

     bvhTransform->joint[jID].pos2D[0]=bvhTransform->joint[jID].pos2D[0]-r2DX;
     bvhTransform->joint[jID].pos2D[1]=bvhTransform->joint[jID].pos2D[1]-r2DY;
    }
  }

  return 0;
}
