#ifndef BVH_PROJECT_H_INCLUDED
#define BVH_PROJECT_H_INCLUDED

#include "../bvh_loader.h"
#include "bvh_transform.h"
#include "../mathLibrary.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Configuration parameters for the BVH renderer.
 *
 * The BVH_RendererConfiguration structure holds the configuration parameters for the BVH renderer. It defines the rendering settings such as width, height, frustum limits, intrinsics, distortion, projection matrix, view matrix, and viewport.
 */
struct BVH_RendererConfiguration
{
  int isDefined;
  unsigned int width;
  unsigned int height;

  //Frustrum limits
  float near,far;

  //Intrinsics
  float fX,fY,cX,cY;
  //Distortion
  float k1,k2,k3,p1,p2;
  //----------
  //float R[9];
  float T[4];
  struct Matrix4x4OfFloats projection;
  struct Matrix4x4OfFloats viewMatrix;
  int viewport[4];

  // Per-frame intrinsic randomization.
  // When randomizeIntrinsicsPerFrame is set, dumpBVHTo_JSON_SVG_CSV picks a
  // fresh (fX,fY) from [fXRandomMin,fXRandomMax] x [fYRandomMin,fYRandomMax]
  // for every frame that passes the skeleton filter, rebuilds the renderer,
  // re-projects the frame with the new intrinsics, and writes one row to
  // intrinsics_<tag>.csv alongside the other output CSV files.
  unsigned int randomizeIntrinsicsPerFrame;
  float fXRandomMin, fXRandomMax;
  float fYRandomMin, fYRandomMax;
};

/**
 * @brief Cleans and initializes the BVH transform structure.
 *
 * This function cleans and initializes the BVH transform structure by allocating memory and setting initial values. It prepares the structure for further transformation calculations.
 *
 * @param mc             Pointer to the BVH_MotionCapture structure.
 * @param bvhTransform   Pointer to the BVH_Transform structure to be cleaned and initialized.
 */
void bvh_cleanTransform(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform     * bvhTransform
                      );

/**
 * @brief Projects a specific BVH joint onto the 2D screen space.
 *
 * This function projects a specified BVH joint from 3D world coordinates to 2D screen space. It uses the given BVH motion capture data, transformation data, and renderer configuration to perform the projection.
 *
 * @param mc              The BVH motion capture data.
 * @param bvhTransform    The BVH transformation data.
 * @param renderer        The renderer used for projection.
 * @param jID             The BVH joint ID to project.
 * @param occlusions      The number of occlusions.
 * @param directRendering Flag indicating if direct rendering should be used.
 *
 * @return 1 if the joint was successfully projected, 0 otherwise.
 */
int bvh_projectJIDTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     BVHJointID jID,
                     unsigned int               occlusions,
                     unsigned int               directRendering
                   );

/**
 * @brief Projects the BVH motion capture data onto the 2D screen space.
 *
 * This function projects the BVH motion capture data onto the 2D screen space using the provided transformation and renderer. It handles occlusions if specified and updates the transformation data accordingly.
 *
 * @param mc              The BVH motion capture data.
 * @param bvhTransform    The BVH transformation data.
 * @param renderer        The renderer used for projection.
 * @param occlusions      The number of occlusions.
 * @param directRendering Flag indicating if direct rendering should be used.
 *
 * @return 1 if the projection was successful, 0 otherwise.
 */
int bvh_projectTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     unsigned int               occlusions,
                     unsigned int               directRendering
                   );


#ifdef __cplusplus
}
#endif

#endif // BVH_PROJECT_H_INCLUDED
