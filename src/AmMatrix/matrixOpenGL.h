

#ifndef MATRIXOPENGL_H_INCLUDED
#define MATRIXOPENGL_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include "matrix4x4Tools.h"


/**
* @brief build Projection Matrix using Rodriguez Rotation and a translation ( typically coming from OpenCV )
* @ingroup AmMatrix
* @param  Output Array 4x4 of resulting matrix
* @param  Input Rodriguez Coordinates
* @param  Input Translation Coordinate
* @param  Input Unit Scale
* @retval 0=Failure,1=Success
*/
int convertRodriguezAndTranslationTo4x4UnprojectionMatrix(float * result4x4, float * rodriguez , float * translation , float scaleToDepthUnit);

/**
* @brief build OpenGL Projection Matrix using Rodriguez Rotation and a translation ( typically coming from OpenCV )
* @ingroup AmMatrix
* @param  Output Array 4x4 of resulting matrix
* @param  Input Rodriguez Coordinates
* @param  Input Translation Coordinate
* @param  Input Unit Scale
* @retval 0=Failure,1=Success
*/
int convertRodriguezAndTranslationToOpenGL4x4ProjectionMatrix(float * result4x4, float * rodriguez , float * translation , float scaleToDepthUnit);



/**
* @brief build OpenGL Projection Matrix simulating a "real" camera
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array 3x3 of object Rotation
* @param  Input Array 4x1 of absolute 3D position of the point ( X,Y,Z,W )
* @retval 0=Failure,1=Success
*/
void buildOpenGLProjectionForIntrinsics   (
                                             float * frustum,
                                             int * viewport ,
                                             float fx,
                                             float fy,
                                             float skew,
                                             float cx, float cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             float nearPlane,
                                             float farPlane
                                           );


/**
* @brief build OpenGL Projection Matrix simulating a "real" camera
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array 3x3 of object Rotation
* @param  Input Array 4x1 of absolute 3D position of the point ( X,Y,Z,W )
* @retval 0=Failure,1=Success
*/
void buildOpenGLProjectionForIntrinsics_OpenGLColumnMajor(
                                             float * frustum,
                                             int * viewport ,
                                             float fx,
                                             float fy,
                                             float skew,
                                             float cx,float cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             float nearPlane,
                                             float farPlane
                                           );






/**
* @brief Calculate an OpenGL frustrum matrix using floats
* @ingroup OGLTools
* @param Output Matrix
* @param Left Limit
* @param Right Limit
* @param Bottom Limit
* @param Top Limit
* @param Z Near
* @param Z Far
*/
void glhFrustumf2(
                  float *matrix,
                  float left,
                  float right,
                  float bottom,
                  float top,
                  float znear,
                  float zfar
                 );




/**
* @brief Calculate an OpenGL perspective matrix using floats
* @ingroup OGLTools
* @param Output Matrix
* @param Field Of View in Degrees
* @param Aspect Ratio
* @param Z Near
* @param Z Far
*/
void glhPerspectivef2(
                      float *matrix,
                      float fovyInDegrees,
                      float aspectRatioV,
                      float znear,
                      float zfar
                     );




/**
* @brief Calculate an OpenGL perspective matrix using doubles
* @ingroup OGLTools
* @param Output Matrix
* @param Field Of View in Degrees
* @param Aspect Ratio
* @param Z Near
* @param Z Far
*/
void gldPerspective(
                     float *matrix,
                     float fovxInDegrees,
                     float aspect,
                     float zNear,
                     float zFar
                   );








  int _glhProjectf(float * position3D, float *modelview, float *projection, int *viewport, float *windowCoordinate);


  int _glhUnProjectf(float winx, float winy, float winz, float *modelview, float *projection, int *viewport, float *objectCoordinate);



  void glGetViewportMatrix(float * m, float startX, float startY, float width, float height, float near, float far);



  void getModelViewProjectionMatrixFromMatrices(struct Matrix4x4OfFloats * output,struct Matrix4x4OfFloats * projectionMatrix,struct Matrix4x4OfFloats * viewMatrix,struct Matrix4x4OfFloats * modelMatrix);


  void prepareRenderingMatrices(
                              float fx ,
                              float fy ,
                              float skew ,
                              float cx,
                              float cy,
                              float windowWidth,
                              float windowHeight,
                              float near,
                              float far,
                              struct Matrix4x4OfFloats * projectionMatrix,
                              struct Matrix4x4OfFloats * viewMatrix,
                              struct Matrix4x4OfFloats * viewportMatrix
                             );


void correctProjectionMatrixForDifferentViewport(
                                                  float * out,
                                                  float * projectionMatrix,
                                                  float * originalViewport,
                                                  float * newViewport
                                                );

#ifdef __cplusplus
}
#endif


#endif //MATRIXOPENGL_H_INCLUDED
