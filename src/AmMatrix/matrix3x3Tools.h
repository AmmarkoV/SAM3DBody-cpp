/** @file matrix3x3Tools.h
 *  @brief  A small 3x3 matrix library for simple 3x3 transformations
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIX3X3TOOLS_H_INCLUDED
#define MATRIX3X3TOOLS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


/**
* @brief Allocate a new 3x3 Matrix
* @ingroup AmMatrix
* @retval 0=Failure or a pointer to an allocated 3x3 Matrix
*/
double * alloc3x3Matrix();


/**
* @brief Deallocate an existing 3x3 Matrix
* @ingroup AmMatrix
* @param  Pointer to a Pointer of an allocated matrix
*/
void free3x3Matrix(double ** mat);

/**
* @brief Printout an 3x3 Matrix that consists of floats
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated float matrix
*/
void print3x3FMatrix(const char * str , float * matrix3x3);

/**
* @brief Printout an 3x3 Matrix that consists of doubles
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void print3x3DMatrix(const char * str , double * matrix3x3);

/**
* @brief Printout an 3x3 Matrix that consists of doubles in Scilab Friendly mode
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void print3x3DScilabMatrix(const char * str , double * matrix3x3);

/**
* @brief Copy a 3x3 Matrix to another
* @ingroup AmMatrix
* @param  Output Matrix
* @param  Input Matrix
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void copy3x3Matrix(double * out,double * in);

/**
* @brief Set an allocated 3x3 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
*/
void create3x3IdentityMatrix(double * m);


void convert3x3DMatrixto3x3F(float * d, double * m );


/**
* @brief Set an allocated 3x3 matrix to contain a rotation (  )
* @ingroup AmMatrix
* @param   Output 3x3 Matrix
* @param   Input angle X
* @param   Input angle Y
* @param   Input angle Z
*/
void create3x3MatrixFromEulerAnglesXYZ(double * m ,double x, double y, double z);


/**
* @brief Set an allocated 3x3 matrix to contain a rotation (  )
* @ingroup AmMatrix
* @param   Output 3x3 Matrix
* @param   Input Axis Unit for the rotation
* @param   Input angle of the Rotation Matrix in Degrees
*/
void  create3x3EulerVectorRotationMatrix(double * matrix3x3,double * axisXYZ,double angle);


/**
* @brief Set an allocated 3x3 matrix to contain a rotation (  )
* @ingroup AmMatrix
* @param   Output 3x3 Matrix
* @param   Input Rotations array in degrees , should be  roll , heading , pitch
*/
void  create3x3EulerRotationXYZOrthonormalMatrix(double * matrix3x3,double * rotationsXYZ);

/**
* @brief Transpose an allocated 3x3 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
* @retval 0=Failure,1=Success
*/
int transpose3x3MatrixD(double * mat);



/**
* @brief Transpose an allocated 3x3 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Output Matrix
* @param  Input Matrix
* @retval 0=Failure,1=Success
*/
int transpose3x3MatrixDFromSource(double * dest,double * source);

/**
* @brief Convert an allocated 3x3 matrix to an allocated 4x4 Matrix
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Input 3x3 Matrix
* @retval 0=Failure,1=Success
*/
int upscale3x3Fto4x4F(float * mat4x4,float * mat3x3);


/**
* @brief Replace all items of the matrix with random values of a given range
* @ingroup AmMatrix
* @param  Input/Output 3x3 Matrix
* @param  Minimum Values of Items
* @param  Maximum Values of Items
* @retval 0=Failure,1=Success
*/
int random3x3Matrix(double * mat,double minimumValues, double maximumValues);



/**
* @brief Compute the determenant of a 3x3 matrix
* @ingroup AmMatrix
* @param  Input 3x3 Matrix
* @retval Det(mat)
*/
double det3x3Matrix(double * mat);

/**
* @brief Invert a 3x3 matrix
* @ingroup AmMatrix
* @param  Input 3x3 Matrix
* @param  Output ( should be already allocated ) 3x3 Matrix
* @retval 0=failure,1=success
*/
int invert3x3MatrixD(double * mat,double * result);

/**
* @brief Multiply 2x 3x3 matrices ( A * B )
* @ingroup AmMatrix
* @param  Output 3x3 Matrix ( should be already allocated )
* @param  Input 3x3 Matrix A
* @param  Input 3x3 Matrix B
* @retval 0=failure,1=success
*/
int multiplyTwo3x3Matrices(double * result , double * matrixA , double * matrixB);


/**
* @brief Multiply a 3x3 matrix with a Vector (2D Point)  A*V
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 3x3 Matrix A
* @param  Input Vector 3x1 V
* @retval 0=failure,1=success
*/
int transform2DPointVectorUsing3x3Matrix(double * resultPoint2D, double * transformation3x3, double * point2D);


/**
* @brief Normalize a 3x1 matrix with a Vector (2D Point)
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
int normalize2DPointVector(double * vec);


#ifdef __cplusplus
}
#endif

#endif // MATRIX3X3TOOLS_H_INCLUDED
