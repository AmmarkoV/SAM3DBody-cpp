/** @file quaternions.h
 *  @brief This is a small math library that deals with quaternions , floats are used 
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef QUATERNIONS_H_INCLUDED
#define QUATERNIONS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief This enumerator defines the order of quaternions qWqXqYqZ means W X Y Z , and qXqYqZqW means X Y Z W
 */
enum quatOrder
{
  qWqXqYqZ=0,
  qXqYqZqW
};


enum quatOrderXYZW
{
  pQX=0,
  pQY,
  pQZ,
  pQW
};



/**
 * @brief This function converts euler angles to quaternions
 * @ingroup quaternions
 * @param Quaternions , The output quaternions in the order declared by enum quatOrder
 * @param Euler angles , The input euler angles X,Y,Z
 * @param The convention used for quaternions ( see enum quatOrder )
 * @retval Nothing , no return value
 */
void euler2Quaternions(float * quaternions,float * euler,int quaternionConvention);


/**
 * @brief This function converts quaternions to euler angles
 * @ingroup quaternions
 * @param Euler angles , The output euler angles X,Y,Z
 * @param Quaternions , The input quaternions in the order declared by enum quatOrder
 * @param The convention used for quaternions ( see enum quatOrder )
 */
void quaternions2Euler(float * euler,float * quaternions,int quaternionConvention);


/**
 * @brief Perform Slerp function ( mix , smooth ) 2 quaternions
 * @ingroup quaternions
 * @param Output Quaternion
 * @param Input Quaternion A
 * @param Input Quaternion B
 * @param Factor , typically should be 0.5 for half and half
 */
void quaternionSlerp(float * qOut, float * q0,float * q1,float t);

/**
 * @brief Normalize Quaternion
 * @ingroup quaternions
 * @param
 * @param Input/Output , qX
 * @param Input/Output , qY
 * @param Input/Output , qZ
 * @param Input/Output , qW
 * @retval 1=Success/0=Failure
 */
int normalizeQuaternions(float *qX,float *qY,float *qZ,float *qW);

/**
 * @brief Convert Quaternion to a 3x3 Matrix
 * @ingroup quaternions
 * @param
 * @param Output 3x3 float matrix
 * @param Input quaternion
 * @param Input quaternion convention used
 */
void quaternion2Matrix3x3(float * matrix3x3,float * quaternions,int quaternionConvention);

/**
 * @brief Convert Quaternion to a 4x4 Matrix
 * @ingroup quaternions
 * @param
 * @param Output 4x4 float matrix
 * @param Input quaternion
 * @param Input quaternion convention used
 */
void quaternion2Matrix4x4(float * matrix4x4,float * quaternions,int quaternionConvention);



void matrix4x42Quaternion(float * quaternions,int quaternionConvention,float * matrix4x4);
void matrix3x32Quaternion(float * quaternions,int quaternionConvention,float * m3);

/**
 * @brief Calculate the Inner Product of Two Quaternions
 * @ingroup quaternions
 --------------------------
 * @param Quaternion A qX
 * @param Quaternion A qY
 * @param Quaternion A qZ
 * @param Quaternion A qW
 --------------------------
 * @param Quaternion B qX
 * @param Quaternion B qY
 * @param Quaternion B qZ
 * @param Quaternion B qW
 * @retval Inner Product
 */
float innerProductQuaternions(float qAX,float qAY,float qAZ,float qAW ,
                               float qBX,float qBY,float qBZ,float qBW);



/**
 * @brief Calculate the Angle Between Two Quaternions
 * @ingroup quaternions
 --------------------------
 * @param Quaternion A qX
 * @param Quaternion A qY
 * @param Quaternion A qZ
 * @param Quaternion A qW
 --------------------------
 * @param Quaternion B qX
 * @param Quaternion B qY
 * @param Quaternion B qZ
 * @param Quaternion B qW
 * @retval Angle Between the Two
 */
float anglesBetweenQuaternions(float qAX,float qAY,float qAZ,float qAW ,
                                float qBX,float qBY,float qBZ,float qBW);



void quaternionRotate(float * quaternion , float rotX , float rotY, float rotZ , float angleDegrees , int quaternionConvention);


void quaternionFromTwoVectors(float * quaternionOutput , float * vA , float * vB);


void generateRandomQuaternion(float * quaternionOutput);

void stochasticRandomQuaternionWithLessThanAngleDistance(float * quaternionOutput,float * quaternionInput,int quaternionConvention,float angleDistance);

#ifdef __cplusplus
}
#endif


#endif // QUATERNIONS_H_INCLUDED
