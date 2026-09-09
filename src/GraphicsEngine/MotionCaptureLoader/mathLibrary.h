/** @file mathLibrary.h
 *  @brief  BVH files require 4x4 matrix math in order to calculate transforms etc.
 *  This BVH library uses https://github.com/AmmarkoV/RGBDAcquisition/tree/master/tools/AmMatrix 
 *  which should be part of the same repository. In order to simplify porting all includes to the library are 
 *  added here in a central place where paths can be easily modified..!
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef BVH_MATH_LIBRARY_CENTRAL_INCLUDE
#define BVH_MATH_LIBRARY_CENTRAL_INCLUDE


#ifdef __cplusplus
extern "C"
{
#endif

 
#include "simpleRenderer.h"
#include "quaternions.h"
#include "matrixCalculations.h"
#include "matrix4x4Tools.h"


#ifdef __cplusplus
}
#endif




#endif // BVH_MATH_LIBRARY_CENTRAL_INCLUDE
