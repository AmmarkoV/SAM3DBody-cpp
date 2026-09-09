#include "simpleRenderer.h"
#include "matrix4x4Tools.h"
#include "matrixOpenGL.h"
#include <stdio.h>


int simpleRendererDefaults(
                            struct simpleRenderer * sr,
                            unsigned int width,
                            unsigned int height,
                            float fX,
                            float fY
                            )
{
  sr->width=width;
  sr->height=height;
  sr->fx = fX;
  sr->fy = fY;
  sr->skew = 0.0;
  sr->cx = (float) width/2;
  sr->cy = (float) height/2;
  sr->near = 1.0;
  sr->far = 10000.0;

  //-----------------------------------
  sr->cameraOffsetPosition[0]=0.0;
  sr->cameraOffsetPosition[1]=0.0;
  sr->cameraOffsetPosition[2]=0.0;
  sr->cameraOffsetPosition[3]=0.0;
  //-----------------------------------
  sr->cameraOffsetRotation[0]=0.0;
  sr->cameraOffsetRotation[1]=0.0;
  sr->cameraOffsetRotation[2]=0.0;
  sr->cameraOffsetRotation[3]=0.0;
  //-----------------------------------
  return 1;
}


int simpleRendererUpdateMovelViewTransform(struct simpleRenderer * sr)
{
 struct Matrix4x4OfFloats modelTransformationF;
 ///--------------------------------------------------------------------
 ///                       CAMERA MATRICES ETC
 ///--------------------------------------------------------------------
 create4x4FModelTransformation(
                              &modelTransformationF,
                              //Rotation Component
                              (float) sr->cameraOffsetRotation[0],//heading,
                              (float) sr->cameraOffsetRotation[1],//pitch,
                              (float) sr->cameraOffsetRotation[2],//roll,
                              ROTATION_ORDER_RPY,
                              //Translation Component
                              (float) sr->cameraOffsetPosition[0],
                              (float) sr->cameraOffsetPosition[1],
                              (float) sr->cameraOffsetPosition[2],
                              //Scale Component
                              (float) 1.0,
                              (float) 1.0,
                              (float) 1.0
                             );
 ///--------------------------------------------------------------------
 multiplyTwo4x4FMatricesS(&sr->modelViewMatrix,&sr->viewMatrix,&modelTransformationF);
 ///--------------------------------------------------------------------
 return 1;
}




int simpleRendererInitialize(struct simpleRenderer * sr)
{
  sr->viewport[0]=0;
  sr->viewport[1]=0;
  sr->viewport[2]=sr->width;
  sr->viewport[3]=sr->height;

  buildOpenGLProjectionForIntrinsics(
                                      sr->projectionMatrix.m ,
                                      sr->viewport ,
                                      sr->fx,
                                      sr->fy,
                                      sr->skew,
                                      sr->cx,
                                      sr->cy,
                                      sr->width,
                                      sr->height,
                                      sr->near,
                                      sr->far
                                     );

   create4x4FIdentityMatrix(&sr->viewMatrix);
   create4x4FScalingMatrix(&sr->viewMatrix,0.01,0.01,-0.01);

   //Initialization of matrices not yet used
   create4x4FIdentityMatrix(&sr->modelMatrix);
   simpleRendererUpdateMovelViewTransform(sr);    //create4x4FIdentityMatrix(&sr->modelViewMatrix);


 return 1;
}








int simpleRendererInitializeFromExplicitConfiguration(struct simpleRenderer * sr)
{
  //We basically have to do nothing to initialize this way since we have an explicit configuration populating our struct
  //This call is made to basically track when we try to use an explicit configuration
  return 1;
}















int simpleRendererRenderEx(
                         struct simpleRenderer * sr ,
                         float * position3D,
                         float * center3D,
                         float * objectRotation,
                         unsigned int rotationOrder,
                         ///---------------
                         float * output2DX,
                         float * output2DY,
                         float * output2DW,
                         ///---------------
                         char updateModelView
                        )
{
 int res = 0;
 ///--------------------------------------------------------------------
 if (updateModelView)
  { simpleRendererUpdateMovelViewTransform(sr); }
 ///--------------------------------------------------------------------



 ///--------------------------------------------------------------------
 ///                       OBJECT MATRICES ETC
 ///--------------------------------------------------------------------
  struct Matrix4x4OfFloats objectMatrixRotation;

  if (objectRotation==0)
    {
      create4x4FIdentityMatrix(&objectMatrixRotation);
    }   else
    {
     create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                      &objectMatrixRotation ,
                                                      (float) objectRotation[0],
                                                      (float) objectRotation[1],
                                                      (float) objectRotation[2],
                                                      rotationOrder
                                                     );
    }

 ///--------------------------------------------------------------------



   struct Vector4x1OfFloats point3D={0};
   struct Vector4x1OfFloats resultPoint3D={0};
   point3D.m[0]=(float) (position3D[0]-center3D[0]);
   point3D.m[1]=(float) (position3D[1]-center3D[1]);
   point3D.m[2]=(float) (position3D[2]-center3D[2]);
   point3D.m[3]=(float) (1.0);

   transform3DPointFVectorUsing4x4FMatrix(
                                          &resultPoint3D,
                                          &objectMatrixRotation,
                                          &point3D
                                         );


  float final3DPosition[4];
  float windowCoordinates[3]={0};


  if (!sr->removeObjectPosition)
   { //This is more probable to happen
    final3DPosition[0]=(float) resultPoint3D.m[0]+center3D[0]+sr->cameraOffsetPosition[0];
    final3DPosition[1]=(float) resultPoint3D.m[1]+center3D[1]+sr->cameraOffsetPosition[1];
    final3DPosition[2]=(float) resultPoint3D.m[2]+center3D[2]+sr->cameraOffsetPosition[2];
    final3DPosition[3]=(float) 1.0;//resultPoint3D.m[3];
   } else
   {
    final3DPosition[0]=(float) resultPoint3D.m[0]+sr->cameraOffsetPosition[0];
    final3DPosition[1]=(float) resultPoint3D.m[1]+sr->cameraOffsetPosition[1];
    final3DPosition[2]=(float) resultPoint3D.m[2]+sr->cameraOffsetPosition[2];
    final3DPosition[3]=(float) 1.0;//resultPoint3D.m[3];
   }

 ///--------------------------------------------------------------------




 ///--------------------------------------------------------------------
 ///                         FINAL PROJECTION
 ///--------------------------------------------------------------------
  if (
       _glhProjectf(
                     final3DPosition,
                     sr->modelViewMatrix.m,
                     sr->projectionMatrix.m,
                     sr->viewport,
                     windowCoordinates
                    )
     )
     {
      *output2DX = windowCoordinates[0];//windowCoordinates[2];
      *output2DY = windowCoordinates[1];//windowCoordinates[2];
      *output2DW = windowCoordinates[2];
      res = 1;
     }
     // else
     //{
        //If you reach here make sure you have called simpleRendererInitialize
        /*
        print4x4FMatrix("modelViewMatrix",sr->modelMatrix,1);
        print4x4FMatrix("projectionMatrix",sr->projectionMatrix,1);
        fprintf(stderr,"simpleRendererRender: Could not project 3D Point (%0.2f,%0.2f,%0.2f)\n",final3DPosition[0],final3DPosition[1],final3DPosition[2]);
        */
    // }
 ///--------------------------------------------------------------------

 return res;
}





int simpleRendererRender(
                         struct simpleRenderer * sr ,
                         float * position3D,
                         float * center3D,
                         float * objectRotation,
                         unsigned int rotationOrder,
                         ///---------------
                         float * output2DX,
                         float * output2DY,
                         float * output2DW
                        )
{
   return simpleRendererRenderEx(
                                 sr,
                                 position3D,
                                 center3D,
                                 objectRotation,
                                 rotationOrder,
                                 ///---------------
                                 output2DX,
                                 output2DY,
                                 output2DW,
                                 ///---------------
                                 1 //Do updates..!
                                );
}


int simpleRendererRenderUsingPrecalculatedMatrices(
                                                    struct simpleRenderer * sr ,
                                                    float * position3D,
                                                    ///---------------
                                                    float * output2DX,
                                                    float * output2DY,
                                                    float * output2DW
                                                  )
{
 ///--------------------------------------------------------------------
 ///                         FINAL PROJECTION
 ///--------------------------------------------------------------------
  float windowCoordinates[3]={0};

  create4x4FIdentityMatrix(&sr->modelViewMatrix);

  if (
       !_glhProjectf(
                     position3D,
                     sr->modelViewMatrix.m,
                     sr->projectionMatrix.m,
                     sr->viewport,
                     windowCoordinates
                    )
     )
     {
         //fprintf(stderr,"simpleRendererRenderUsingPrecalculatedMatrices: Could not project 3D Point (%0.2f,%0.2f,%0.2f)\n",position3D[0],position3D[1],position3D[2]);
     }
 ///--------------------------------------------------------------------
  *output2DX = windowCoordinates[0];//windowCoordinates[2];
  *output2DY = windowCoordinates[1];//windowCoordinates[2];
  *output2DW = windowCoordinates[2];
  return 1;
}




//TODO : I should add simpleRendererRenderUsingPrecalculatedMatrices from vectors to cut down on extensive function calling and make the code more efficient
int simpleRendererMultiRenderUsingPrecalculatedMatrices(
                                                          struct simpleRenderer * sr ,
                                                          float * position3DVector,
                                                          unsigned int numberOf3DPoints,
                                                          ///---------------
                                                          float * output2DX,
                                                          float * output2DY,
                                                          float * output2DW
                                                        )
{

  return 0;
}




