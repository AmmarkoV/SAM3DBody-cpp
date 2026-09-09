/** @file simpleRenderer.h
 *  @ingroup simpleRenderer
 *  @brief This is a simple 3D renderer that can be setup to mirror a virtual camera and convert 3D points to their 2D projections in a virtual frame.
 *  Please be advised that there is no rasterization being performed,  there is no depth ordering and the output not a 2D frame but rather 2D points.
 *  So this code is not suitable as a software renderer for 3D graphics but rather as a tool to be able to perform 2D comparisons over an RGB frame
 *  using 3D models and 3D point clouds.
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef SIMPLERENDERER_H_INCLUDED
#define SIMPLERENDERER_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include "matrix4x4Tools.h"


/**
 * @brief This structure holds the 3D renderer configuration, if you want to initialize this structure please use simpleRendererDefaults
 * @ingroup simpleRenderer
 */
struct simpleRenderer
{
  float fx;
  float fy;
  float skew;
  float cx;
  float cy;
  float near;
  float far;
  float width;
  float height;

  float cameraOffsetPosition[4];
  float cameraOffsetRotation[4];
  int removeObjectPosition;


  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewMatrix;
  struct Matrix4x4OfFloats modelMatrix;
  struct Matrix4x4OfFloats modelViewMatrix;
  int   viewport[4];
};



/**
 * @brief This function converts euler angles to quaternions
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to write to
 * @param The width of the virtual frame in pixels
 * @param The height of the virtual frame in pixels
 * @param The focal length for the virtual frame in the X dimension
 * @param The focal length for the virtual frame in the Y dimension
 * @retval 1=Success/0=Failure
 */
int simpleRendererDefaults(
                            struct simpleRenderer * sr,
                            unsigned int width,
                            unsigned int height,
                            float fX,
                            float fY
                          );



/**
 * @brief This function initializes a simple renderer instance to become ready to render 3D points. The simpleRendererDefaults must be called before this call to make sure the
 *  new renderer is not initialized using wrong settings
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to initialize
 * @retval 1=Success/0=Failure
 */
int simpleRendererInitialize(struct simpleRenderer * sr);


/**
 * @brief A *dangerous* way to initialize the renderer assuming the simpleRenderer struct is already populated with correct values, This function basically does nothing
 * acts like a placeholder
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to initialize
 * @retval 1=Success/0=Failure
 */
int simpleRendererInitializeFromExplicitConfiguration(struct simpleRenderer * sr);


/**
 * @brief This is a faster version of the simpleRendererRenderer function that assumes that  modelViewMatrix, projectionMatrix and the viewport has been already calculated
 * so it can get away with less calculations and allow more efficient execution
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to render with
 * @param The 3D position of the point we want to transform
 * @param Output 2D pixel in the X axis
 * @param Output 2D pixel in the Y axis
 * @param Output 2D pixel normalization

 * @retval 1=Success/0=Failure
 */
int simpleRendererRenderUsingPrecalculatedMatrices(
                                                    struct simpleRenderer * sr ,
                                                    float * position3D,
                                                    ///---------------
                                                    float * output2DX,
                                                    float * output2DY,
                                                    float * output2DW
                                                  );



int simpleRendererUpdateMovelViewTransform(struct simpleRenderer * sr);





/**
 * @brief This is the core function of simpleRenderer that converts a 3D position to a 2D projection based on the configuration of the simpleRenderer, its position, rotation and center
 * of reference
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to render with
 * @param The 3D position of the point we want to transform
 * @param The center of reference of our 3D point ( cannot be null, you will need to fill it with zeros if you dont want to use it )
 * @param The rotation of the transform based on the camera
 * @param The rotation order of above stated rotation
 * @param Output 2D pixel in the X axis
 * @param Output 2D pixel in the Y axis
 * @param Output 2D pixel normalization
 * @param Update ModelView Matrix

 * @retval 1=Success/0=Failure
 */
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
                        );



/**
 * @brief This is the core function of simpleRenderer that converts a 3D position to a 2D projection based on the configuration of the simpleRenderer, its position, rotation and center
 * of reference
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to render with
 * @param The 3D position of the point we want to transform
 * @param The center of reference of our 3D point ( cannot be null, you will need to fill it with zeros if you dont want to use it )
 * @param The rotation of the transform based on the camera
 * @param The rotation order of above stated rotation
 * @param Output 2D pixel in the X axis
 * @param Output 2D pixel in the Y axis
 * @param Output 2D pixel normalization

 * @retval 1=Success/0=Failure
 */
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
                        );





/**
 * @brief This function is a stub, in the future this code will be vectorized to enable faster execution
 * @ingroup simpleRenderer
 * @param The simpleRenderer structure that we want to render with
 * @param A vector to the 3D positions we want to transform
 * @param Number of 3D positions on our input vector
 * @param Output 2D pixel in the X axis
 * @param Output 2D pixel in the Y axis
 * @param Output 2D pixel normalization

 * @retval 1=Success/0=Failure
 */
int simpleRendererMultiRenderUsingPrecalculatedMatrices(
                                                          struct simpleRenderer * sr ,
                                                          float * position3DVector,
                                                          unsigned int numberOf3DPoints,
                                                          ///---------------
                                                          float * output2DX,
                                                          float * output2DY,
                                                          float * output2DW
                                                        );

#ifdef __cplusplus
}
#endif


#endif // SIMPLERENDERER_H_INCLUDED
