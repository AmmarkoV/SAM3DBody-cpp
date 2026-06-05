/** @file matrix4x4Tools.h
 *  @brief  A small 4x4 matrix library for simple 4x4 transformations
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIX4X4TOOLS_H_INCLUDED
#define MATRIX4X4TOOLS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

int codeHasSSE();
int codeHasAVX2();

//This should be ROTATION_ORDER_NAMESA but it isn't to avoid bugs
//Since this used to be a variable in some points of the code..
static const char * ROTATION_ORDER_NAMESA[] =
{
  "ROTATION_ORDER_NONE", //0
  "ROTATION_ORDER_XYZ",//1
  "ROTATION_ORDER_XZY",//2
  "ROTATION_ORDER_YXZ",//3
  "ROTATION_ORDER_YZX",//4
  "ROTATION_ORDER_ZXY",//5
  "ROTATION_ORDER_ZYX",//6
  "ROTATION_ORDER_RPY",//7
  "ROTATION_ORDER_RODRIGUES",//8
  //--------------------
  "INVALID_ROTATION_ORDER"
};


enum ROTATION_ORDER
{
  ROTATION_ORDER_NONE=0,
  ROTATION_ORDER_XYZ,//1
  ROTATION_ORDER_XZY,//2
  ROTATION_ORDER_YXZ,//3
  ROTATION_ORDER_YZX,//4
  ROTATION_ORDER_ZXY,//5
  ROTATION_ORDER_ZYX,//6
  ROTATION_ORDER_RPY,//7
  ROTATION_ORDER_RODRIGUES,//8
  //--------------------
  ROTATION_ORDER_NUMBER_OF_NAMES
};


struct Matrix4x4OfFloats
{
  /*A Matrix 4x4 aligned to allow for SSE optimized calculations.
   *
   * Items are stored on the m array using this ordering
     0   1   2   3
     4   5   6   7
     8   9   10  11
     12  13  14  15

     IRC => Item Row/Column =>
     I11     , I12 , I13 , I14 ,
     I21     , I22 , I23 , I24 ,
     I31     , I32 , I33 , I34 ,
     I41     , I42 , I43 , I44
    */
  float __attribute__((aligned(16))) m[16];
};


struct Vector4x1OfFloats
{
  /*A Matrix 4x1 aligned to allow for SSE optimized calculations.
   *
   * Items are stored on the m array using this ordering
     0   1   2   3

     IRC => Item Row/Column =>
     I11, I12, I13, I14
    */
  float __attribute__((aligned(16))) m[4];
};


float degrees_to_radF(float degrees);

/**
* @brief Printout an 4x4 Matrix that consists of floats
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated float matrix
*/
void print4x4FMatrix(const char * str , float * matrix4x4,int forcePrint);

/**
* @brief Printout an 4x4 Matrix that consists of doubles
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void print4x4DMatrix(const char * str , double * matrix4x4,int forcePrint);

/**
* @brief Copy a 4x4 Matrix of doubles to another
* @ingroup AmMatrix
* @param  Output Matrix
* @param  Input Matrix
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void copy4x4DMatrix(double * out,double * in);


void copy3x3FMatrixTo4x4F(float * out,float * in);


void copy4x4FMatrixToAlignedContainer(struct Matrix4x4OfFloats * out,const float * in);

/**
* @brief Copy a 4x4 Matrix of floats to another
* @ingroup AmMatrix
* @param  Output Matrix
* @param  Input Matrix
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void copy4x4FMatrix(float * out,const float * in);

/**
* @brief Convert a 4x4 Matrix from Float To Double
* @ingroup AmMatrix
* @param  Pointer to a double 4x4 output
* @param  Pointer to a float 4x4 input
*/
void copy4x4FMatrixTo4x4D(double * out,float * in);

/**
* @brief Convert a 4x4 Matrix from Double To Float
* @ingroup AmMatrix
* @param  Pointer to a float 4x4 output
* @param  Pointer to a double 4x4 input
*/
void copy4x4DMatrixTo4x4F(float * dest, double * m );



void create4x4FIdentityMatrixDirect(float * m);

/**
* @brief Set an allocated 4x4 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
*/
void create4x4FIdentityMatrix(struct Matrix4x4OfFloats * m);


int is4x4FIdentityMatrix(float * m);

int is4x4FZeroMatrix(float  * m);


int are4x4FMatricesEqual(float * m1,float * m2);

int is4x4FIdentityMatrixS(struct Matrix4x4OfFloats * m);

int is4x4FIdentityMatrixPercisionCompensating(struct Matrix4x4OfFloats * m);

void convert4x4FMatrixToRPY(struct Matrix4x4OfFloats * m ,float *roll,float *pitch,float *yaw);

/**
* @brief Convert an allocated 4x4 matrix to a homogeneous Translation rotation
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Input angle
* @param  X Axis Parameter
* @param  Y Axis Parameter
* @param  Z Axis Parameter
*/
void create4x4FRotationMatrix(struct Matrix4x4OfFloats * m , float angle, float x, float y, float z);

/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Translation
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  X Translation
* @param  Y Translation
* @param  Z Translation
*/
void create4x4FTranslationMatrix(struct Matrix4x4OfFloats * m , float x, float y, float z);


/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Scaling
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Scaling on the X
* @param  Scaling on the Y
* @param  Scaling on the Z
* @retval 0=Failure,1=Success
*/
void create4x4FScalingMatrix(struct Matrix4x4OfFloats * matrix , float scaleX, float scaleY, float scaleZ);

/**
* @brief Convert a quaternion to 4x4 matrix
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Quaternion on the X
* @param  Quaternion on the Y
* @param  Quaternion on the Z
* @param  Quaternion on the W
* @param  Degrees of rotation
*/
void create4x4FQuaternionMatrix(struct Matrix4x4OfFloats * m ,float qX,float qY,float qZ,float qW);

/**
* @brief Convert euler angles in degrees to a 4x4 rotation matrix using any rotation order wanted
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Rotation X in euler angles (0-360)
* @param  Rotation Y in euler angles (0-360)
* @param  Rotation Z in euler angles (0-360)
* @param  Rotation order given by enum ROTATION_ORDER, typically ROTATION_ORDER_ZYX or ROTATION_ORDER_XYZ
*/
void create4x4FMatrixFromEulerAnglesWithRotationOrder(struct Matrix4x4OfFloats * m,float degreesEulerX, float degreesEulerY, float degreesEulerZ,unsigned int rotationOrder);



void create4x4FMatrixFromEulerAnglesXYZAllInOne(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ);

/**
* @brief Convert a quaternion to 4x4 matrix ZYX convention ( standard )
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Rotation X in euler angles (0-360)
* @param  Rotation Y in euler angles (0-360)
* @param  Rotation Z in euler angles (0-360)
*/
void create4x4FMatrixFromEulerAnglesZYX(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ);


/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Rotation on the X axis
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Degrees of rotation
*/
void create4x4FRotationX(struct Matrix4x4OfFloats * m,float degrees);

/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Rotation on the Y axis
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Degrees of rotation
*/
void create4x4FRotationY(struct Matrix4x4OfFloats * m,float degrees);

/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Rotation on the Z axis
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Degrees of rotation
*/
void create4x4FRotationZ(struct Matrix4x4OfFloats * m,float degrees);


/**
* @brief Compute the determenant of a 4x4 matrix
* @ingroup AmMatrix
* @param  Input 4x4 Matrix
* @retval Det(mat)
*/
float det4x4FMatrix(const float * mat) ;

/**
* @brief Invert a 4x4 matrix
* @ingroup AmMatrix
* @param  Input 4x4 Matrix
* @param  Output ( should be already allocated ) 3x3 Matrix
* @retval 0=failure,1=success
*/
int invert4x4FMatrix(struct Matrix4x4OfFloats * result,const struct Matrix4x4OfFloats * mat) ;


/**
* @brief Transpose an allocated 4x4 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
* @retval 0=Failure,1=Success
*/
int transpose4x4FMatrix(float * mat);



/**
* @brief Transpose an allocated 4x4 matrix to Identity using doubles ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
* @retval 0=Failure,1=Success
*/
int transpose4x4DMatrix(double * mat);



/**
* @brief Multiply 2x 4x4 Double matrices ( A * B )
* @ingroup AmMatrix
* @param  Output 4x4 Float Matrix ( should be already allocated )
* @param  Input 4x4 Float Matrix A
* @param  Input 4x4 Float Matrix B
* @retval 0=failure,1=success
*/
int multiplyTwo4x4DMatrices(double * result ,const double * matrixA ,const double * matrixB);

/**
* @brief Multiply 3x 4x4 Double matrices ( A * B * C )
* @ingroup AmMatrix
* @param  Output 4x4 Float Matrix ( should be already allocated )
* @param  Input 4x4 Float Matrix A
* @param  Input 4x4 Float Matrix B
* @param  Input 4x4 Float Matrix C
* @retval 0=failure,1=success
*/
void multiplyThree4x4DMatrices(double * result ,const double * matrixA ,const double * matrixB ,const double * matrixC);


/**
* @brief Multiply 2x 4x4 Float matrices ( A * B )
* @ingroup AmMatrix
* @param  Output 4x4 Float Matrix ( should be already allocated )
* @param  Input 4x4 Float Matrix A
* @param  Input 4x4 Float Matrix B
* @retval 0=failure,1=success
*/
//static inline void multiplyTwo4x4FMatrices_Naive(float * result ,const float * matrixA ,const float * matrixB);



/**
* @brief Multiply 2x 4x4 Float matrices ( A * B ) using SSE instrcutions,  matrices should be 16bit algined i.e.  float m[16] __attribute__((aligned(16))) )
* @ingroup AmMatrix
* @param  Output 4x4 Float Matrix ( should be already allocated )
* @param  Input 4x4 Float Matrix A
* @param  Input 4x4 Float Matrix B
* @retval 0=failure,1=success
*/
void multiplyTwo4x4FMatricesS(struct Matrix4x4OfFloats * result ,const struct Matrix4x4OfFloats * matrixA ,const struct Matrix4x4OfFloats * matrixB);

void multiplyTwoRaw4x4FMatricesS(float * result ,const float * matrixA ,const float * matrixB);

void multiplyTwo4x4FMatricesBuffered(struct Matrix4x4OfFloats * result,const float * matrixA,const float * matrixB);

void multiplyThree4x4FMatrices(struct Matrix4x4OfFloats * result,const struct Matrix4x4OfFloats * matrixA,const struct Matrix4x4OfFloats * matrixB,const struct Matrix4x4OfFloats * matrixC);

int multiplyThree4x4FMatricesWithIdentityHints(
                                                struct Matrix4x4OfFloats * result,
                                                const struct Matrix4x4OfFloats * matrixA,
                                                int matrixAIsIdentity,
                                                const struct Matrix4x4OfFloats * matrixB,
                                                int matrixBIsIdentity,
                                                const struct Matrix4x4OfFloats * matrixC,
                                                int matrixCIsIdentity
                                              );

void multiplyThree4x4FMatrices_Naive(float * result ,const float * matrixA ,const float * matrixB ,const float * matrixC);

void multiplyFour4x4FMatrices(
                              struct Matrix4x4OfFloats * result,
                              const struct Matrix4x4OfFloats * matrixA,
                              const struct Matrix4x4OfFloats * matrixB,
                              const struct Matrix4x4OfFloats * matrixC,
                              const struct Matrix4x4OfFloats * matrixD
                             );



/**
* @brief Multiply a 4x4 matrix of floats with a float Vector (3D Point)  A*V
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V
* @retval 0=failure,1=success
*/
int transform3DPointFVectorUsing4x4FMatrix_Naive(float * resultPoint3D,float * transformation4x4,float * point3D);


/**
* @brief Multiply a 4x4 matrix of floats with a float Vector (3D Point)  A*V
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V
* @retval 0=failure,1=success
*/
int transform3DPointFVectorUsing4x4FMatrix(struct Vector4x1OfFloats * resultPoint3D,struct Matrix4x4OfFloats * transformation4x4,struct Vector4x1OfFloats * point3D);

/**
* @brief Multiply a the 3x3 rotational part of a 4x4 matrix with a Normal Vector (3D Point)  A*V
         Basically just performing a (3x3) x (3x1) operation from a (4x4) x (4x1) input
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V where W coordinate should be 0
* @retval 0=failure,1=success
*/
int transform3DNormalVectorUsing3x3FPartOf4x4FMatrix(float * resultPoint3D,struct Matrix4x4OfFloats * transformation4x4,float * point3D);



int normalize3DPointFVector(float * vec);

/**
* @brief Normalize a 4x1 matrix with a Vector (3D Point)
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
int normalize3DPointDVector(double * vec);


void doRPYTransformationF(
                         struct Matrix4x4OfFloats * m,
                         float  rollInDegrees,
                         float  pitchInDegrees,
                         float  yawInDegrees
                        );


/*
*/




/**
* @brief Produce a rotation and translation that will bring the scene to the coordinate frame of the camera in order to properly
         render objects..!

          This code produces the same matrix as the one produced by the following openGL calls

          glTranslatef(x,y,z);
          if ( roll!=0 )    { glRotatef(roll,0.0,0.0,1.0); }
          if ( heading!=0 ) { glRotatef(heading,0.0,1.0,0.0); }
          if ( pitch!=0 )   { glRotatef(pitch,1.0,0.0,0.0); }

          if ( (scaleX!=1.0) || (scaleY!=1.0) || (scaleZ!=1.0) )
                       {
                         glScalef( scaleX , scaleY , scaleZ );
                       }
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
void create4x4FModelTransformation(
                                   struct Matrix4x4OfFloats * m ,
                                   //Rotation Component
                                   float rotationX,//heading
                                   float rotationY,//pitch
                                   float rotationZ,//roll
                                   unsigned int rotationOrder,
                                   //Translation Component
                                   float x, float y, float z ,
                                   float scaleX, float scaleY, float scaleZ
                                 );




/**
* @brief Produce a rotation and translation that will bring the scene to the coordinate frame of the camera in order to properly
         render objects..!

          This code produces the same matrix as the one produced by the following openGL calls

          glLoadIdentity();
          glRotatef(rotationX_angleDegrees,-1.0,0,0);
          glRotatef(rotationY_angleDegrees,0,-1.0,0);
          glRotatef(rotationZ_angleDegrees,0,0,-1.0); }
          glTranslatef(-camera_pos_x, -camera_pos_y, -camera_pos_z);
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
void create4x4FCameraModelViewMatrixForRendering(
                                                struct Matrix4x4OfFloats * m ,
                                                //Rotation Component
                                                float rotationX_angleDegrees,
                                                float rotationY_angleDegrees,
                                                float rotationZ_angleDegrees ,
                                                //Translation Component
                                                float translationX_angleDegrees,
                                                float translationY_angleDegrees,
                                                float translationZ_angleDegrees
                                               );


#ifdef __cplusplus
}
#endif


#endif // MATRIX4X4TOOLS_H_INCLUDED
