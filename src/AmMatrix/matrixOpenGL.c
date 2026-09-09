#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrixOpenGL.h"

#include "matrix4x4Tools.h"


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


int convertRodriguezTo3x3(float * result,float * matrix)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }

  float x = matrix[0] , y = matrix[1] , z = matrix[2];
  float th = sqrt( x*x + y*y + z*z );
  float cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  if ( th < 0.00001 )
    {
       result[0]=1.0f;  result[1]=0.0f; result[2]=0.0f;
       result[3]=0.0f;  result[4]=1.0f; result[5]=0.0f;
       result[6]=0.0f;  result[7]=0.0f; result[8]=1.0f;

       //create3x3IdentityMatrix(result);
       return 1;
    }

   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);      result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;          result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;

  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"rodriguez %f %f %f\n ",matrix[0],matrix[1],matrix[2]);
   print3x3FMatrix("Rodriguez Initial", result);
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}


void changeYandZAxisOpenGL4x4Matrix(float * result,float * matrix)
{
  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"Invert Y and Z axis\n");
  #endif // PRINT_MATRIX_DEBUGGING

  struct Matrix4x4OfFloats invertOp = {0};

  create4x4FIdentityMatrix(&invertOp);
  invertOp.m[5]=-1;   invertOp.m[10]=-1;

  multiplyTwoRaw4x4FMatricesS(result,matrix,invertOp.m);
}

int convertRodriguezAndTranslationTo4x4UnprojectionMatrix(float * result4x4, float * rodriguez , float * translation , float scaleToDepthUnit)
{
  float matrix3x3Rotation[16] = {0};

  //Our translation vector is ready to be used!
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);
  #endif // PRINT_MATRIX_DEBUGGING

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((float*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  float * m  = result4x4;
  float * rm = matrix3x3Rotation;
  float * tm = translation;


  //float scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  float Tx = tm[0]*scaleToDepthUnit;
  float Ty = tm[1]*scaleToDepthUnit;
  float Tz = tm[2]*scaleToDepthUnit;


  /*
      Here what we want to do is generate a 4x4 matrix that does the inverse transformation that our
      rodriguez and translation vector define

      In order to do that we should have the following be true

                                      (note the minus under)
      (   R  |  T  )       (   R trans |  - R trans * T  )         (   I  |  0   )
      (  --------- )    .  (  -------------------------- )     =   ( ----------- )
      (   0  |  1  )       (   0       |        1        )         (   0  |  I   )

      Using matlab to do the calculations we get the following matrix
  */

   m[0]=  rm[0];        m[1]= rm[3];        m[2]=  rm[6];       m[3]= -1.0 * ( rm[0]*Tx + rm[3]*Ty + rm[6]*Tz );
   m[4]=  rm[1];        m[5]= rm[4];        m[6]=  rm[7];       m[7]= -1.0 * ( rm[1]*Tx + rm[4]*Ty + rm[7]*Tz );
   m[8]=  rm[2];        m[9]= rm[5];        m[10]= rm[8];       m[11]=-1.0 * ( rm[2]*Tx + rm[5]*Ty + rm[8]*Tz );
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;


  print4x4FMatrix("ModelView", result4x4,0);
  return 1;
}


int convertRodriguezAndTranslationToOpenGL4x4ProjectionMatrix(float * result4x4, float * rodriguez , float * translation , float scaleToDepthUnit )
{
  float matrix3x3Rotation[16] = {0};

  //Our translation vector is ready to be used!
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);
  #endif // PRINT_MATRIX_DEBUGGING


  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((float*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  float * m  = result4x4;
  float * rm = matrix3x3Rotation;
  float * tm = translation;


  //float scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  float Tx = tm[0]*scaleToDepthUnit;
  float Ty = tm[1]*scaleToDepthUnit;
  float Tz = tm[2]*scaleToDepthUnit;

  /*
      Here what we want to do is generate a 4x4 matrix that does the normal transformation that our
      rodriguez and translation vector define
       *
       * //Transposed in one step
  */
   m[0]=  rm[0];        m[4]= rm[1];        m[8]=  rm[2];       m[12]= -Tx;
   m[1]=  rm[3];        m[5]= rm[4];        m[9]=  rm[5];       m[13]= -Ty;
   m[2]=  rm[6];        m[6]= rm[7];        m[10]= rm[8];       m[14]=-Tz;
   m[3]=  0.0;          m[7]= 0.0;          m[11]=0.0;          m[15]=1.0;


  #if PRINT_MATRIX_DEBUGGING
   print4x4FMatrix("ModelView", result4x4);
   fprintf(stderr,"Matrix will be transposed to become OpenGL format ( i.e. column major )\n");
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}



void buildOpenGLProjectionForIntrinsics(
                                         float * frustum,
                                         int   * viewport,
                                         float fx,
                                         float fy,
                                         float skew,
                                         float cx, float cy,
                                         unsigned int imageWidth, unsigned int imageHeight,
                                         float nearPlane,
                                         float farPlane
                                        )
{
   fprintf(stderr,"buildOpenGLProjectionForIntrinsics: img(%ux%u) ",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.2f, fy %0.2f, cx %0.2f, cy %0.2f\n",fx,fy,cx,cy);
   //fprintf(stderr,"skew %0.2f, Near %0.2f, Far %0.2f\n",skew,nearPlane,farPlane);

   if (farPlane==0.0)
   {
    fprintf(stderr,RED "Far plane is zero, argument bug..? \n" NORMAL);
    exit(1);
   }


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    float L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    float N = nearPlane , F = farPlane;
    float R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limits R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0f*fx/R_sub_L;     frustum[1] = 0.0f;                 frustum[2] = 0.0f;                              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0f*fy/T_sub_B;      frustum[6] = 0.0f;                              frustum[7] = 0.0f;
   frustum[8] = 2.0f*cx/R_sub_L-1.0f; frustum[9] = 2.0f*cy/T_sub_B-1.0f; frustum[10]=-1.0*(F_plus_N/F_sub_N);            frustum[11] = -1.0f;
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]=-2.0f*F_mul_N/(F_sub_N);            frustum[15] = 0.0f;
   //Matrix already in OpenGL column major format



   //TROUBLESHOOTING Left To Right Hand conventions , Thanks Damien 24-06-15
   struct Matrix4x4OfFloats identMat={0};
   struct Matrix4x4OfFloats finalFrustrum={0};

   create4x4FIdentityMatrix(&identMat);
   identMat.m[10]=-1;
   multiplyTwoRaw4x4FMatricesS(finalFrustrum.m,identMat.m,frustum);
   copy4x4FMatrix(frustum,finalFrustrum.m);

   //This should produce our own Row Major Format
   transpose4x4FMatrix(finalFrustrum.m);
}


void buildOpenGLProjectionForIntrinsics_OpenGLColumnMajor
                                           (
                                             float * frustumOutput,
                                             int * viewport ,
                                             float fx,
                                             float fy,
                                             float skew,
                                             float cx, float cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             float nearPlane,
                                             float farPlane
                                           )
{
//   fprintf(stderr,"buildOpenGLProjectionForIntrinsicsD according to old Ammar code Image ( %u x %u )\n",imageWidth,imageHeight);
//   fprintf(stderr,"fx %0.2f fy %0.2f , cx %0.2f , cy %0.2f , skew %0.2f \n",fx,fy,cx,cy,skew);
//   fprintf(stderr,"Near %0.2f Far %0.2f \n",nearPlane,farPlane);

   if (farPlane==0.0)
   {
    fprintf(stderr,RED "Far plane is zero, argument bug..? \n" NORMAL);
    exit(1);
   }


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    float L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    float N = nearPlane , F = farPlane;
    float R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limits R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   struct Matrix4x4OfFloats frustum={0};
   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum.m[0] = -2.0f*fx/R_sub_L;     frustum.m[1] = 0.0f;                 frustum.m[2] = 0.0f;                              frustum.m[3]  = 0.0f;
   frustum.m[4] = 0.0f;                 frustum.m[5] = 2.0f*fy/T_sub_B;      frustum.m[6] = 0.0f;                              frustum.m[7]  = 0.0f;
   frustum.m[8] = 2.0f*cx/R_sub_L-1.0f; frustum.m[9] = 2.0f*cy/T_sub_B-1.0f; frustum.m[10]=-1.0*(F_plus_N/F_sub_N);            frustum.m[11] =-1.0f;
   frustum.m[12]= 0.0f;                 frustum.m[13]= 0.0f;                 frustum.m[14]=-2.0f*F_mul_N/(F_sub_N);            frustum.m[15] = 0.0f;
   //Matrix already in OpenGL column major format

   //TROUBLESHOOTING Left To Right Hand conventions , Thanks Damien 24-06-2015
   struct Matrix4x4OfFloats identMat;
   struct Matrix4x4OfFloats finalFrustrum;
   create4x4FIdentityMatrix(&identMat);
   identMat.m[10]=-1;
   multiplyTwo4x4FMatricesS(&finalFrustrum,&identMat,&frustum);
   copy4x4FMatrix(frustumOutput,finalFrustrum.m);
}

//matrix will receive the calculated perspective matrix.
//You would have to upload to your shader
// or use glLoadMatrixf if you aren't using shaders.
void glhFrustumf2(
                  float *matrix,
                  float left,
                  float right,
                  float bottom,
                  float top,
                  float znear,
                  float zfar
                 )
{
    float temp, temp2, temp3, temp4;
    temp = 2.0 * znear;
    temp2 = right - left;
    temp3 = top - bottom;
    temp4 = zfar - znear;
    matrix[0] = temp / temp2;
    matrix[1] = 0.0;
    matrix[2] = 0.0;
    matrix[3] = 0.0;
    matrix[4] = 0.0;
    matrix[5] = temp / temp3;
    matrix[6] = 0.0;
    matrix[7] = 0.0;
    matrix[8] = (right + left) / temp2;
    matrix[9] = (top + bottom) / temp3;
    matrix[10] = (-zfar - znear) / temp4;
    matrix[11] = -1.0;
    matrix[12] = 0.0;
    matrix[13] = 0.0;
    matrix[14] = (-temp * zfar) / temp4;
    matrix[15] = 0.0;
}


void glhPerspectivef2(
                      float *matrix,
                      float fovyInDegrees,
                      float aspectRatioV,
                      float znear,
                      float zfar
                     )
{
    float ymax, xmax;
    ymax = znear * tan(fovyInDegrees * M_PI / 360.0);
    //ymin = -ymax;
    //xmin = -ymax * aspectRatioV;
    xmax = ymax * aspectRatioV;
    glhFrustumf2(matrix, -xmax, xmax, -ymax, ymax, znear, zfar);
}




void gldPerspective(
                     float *matrix,
                     float fovxInDegrees,
                     float aspect,
                     float zNear,
                     float zFar
                   )
{
   // This code is based off the MESA source for gluPerspective
   // *NOTE* This assumes GL_PROJECTION is the current matrix
   float xmin, xmax, ymin, ymax;

   xmax = zNear * tan(fovxInDegrees * M_PI / 360.0);
   xmin = -xmax;

   ymin = xmin / aspect;
   ymax = xmax / aspect;

   // Set up the projection matrix
   matrix[0] = (2.0 * zNear) / (xmax - xmin);
   matrix[1] = 0.0;
   matrix[2] = 0.0;
   matrix[3] = 0.0;
   matrix[4] = 0.0;
   matrix[5] = (2.0 * zNear) / (ymax - ymin);
   matrix[6] = 0.0;
   matrix[7] = 0.0;
   matrix[8] = (xmax + xmin) / (xmax - xmin);
   matrix[9] = (ymax + ymin) / (ymax - ymin);
   matrix[10] = -(zFar + zNear) / (zFar - zNear);
   matrix[11] = -1.0;
   matrix[12] = 0.0;
   matrix[13] = 0.0;
   matrix[14] = -(2.0 * zFar * zNear) / (zFar - zNear);
   matrix[15] = 0.0;

   // Add to current matrix
   //glMultMatrixf(matrix);
}



void lookAt(
             float * matrix ,
             float eyex, float eyey, float eyez,
	         float centerx, float centery, float centerz,
	         float upx, float upy, float upz
	        )
{
   float x[3], y[3], z[3];
   float mag;

   /* Make rotation matrix */

   /* Z vector */
   z[0] = eyex - centerx;
   z[1] = eyey - centery;
   z[2] = eyez - centerz;
   mag = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
   if (mag)
    {
      z[0] /= mag;
      z[1] /= mag;
      z[2] /= mag;
    }

   /* Y vector */
   y[0] = upx;
   y[1] = upy;
   y[2] = upz;

   /* X vector = Y cross Z */
   x[0] = y[1] * z[2] - y[2] * z[1];
   x[1] = -y[0] * z[2] + y[2] * z[0];
   x[2] = y[0] * z[1] - y[1] * z[0];

   /* Recompute Y = Z cross X */
   y[0] = z[1] * x[2] - z[2] * x[1];
   y[1] = -z[0] * x[2] + z[2] * x[0];
   y[2] = z[0] * x[1] - z[1] * x[0];

   /* mpichler, 19950515 */
   /* cross product gives area of parallelogram, which is < 1.0 for
    * non-perpendicular unit-length vectors; so normalize x, y here
    */

   mag = sqrt(  x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
   if (mag)
    {
      x[0] /= mag;
      x[1] /= mag;
      x[2] /= mag;
    }

   mag = sqrt( y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
   if (mag)
    {
      y[0] /= mag;
      y[1] /= mag;
      y[2] /= mag;
    }

   struct Matrix4x4OfFloats initial={0};
   initial.m[0] = x[0]; initial.m[1] = x[1]; initial.m[2] = x[2]; initial.m[3] = 0.0;
   initial.m[4] = y[0]; initial.m[5] = y[1]; initial.m[6] = y[2]; initial.m[7] = 0.0;
   initial.m[8] = z[0]; initial.m[9] = z[1]; initial.m[10]= z[2]; initial.m[11]= 0.0;
   initial.m[12]= 0.0;  initial.m[13]= 0.0;  initial.m[14]= 0.0;  initial.m[15]= 1.0;


   /* Translate Eye to Origin */
   //glTranslatef(-eyex, -eyey, -eyez);
   struct Matrix4x4OfFloats translation={0};
   create4x4FTranslationMatrix(&translation , -eyex, -eyey, -eyez );
   multiplyTwoRaw4x4FMatricesS(matrix,initial.m,translation.m);

}




  void MultiplyMatrices4by4OpenGL_FLOAT(float *result, float *matrix1, float *matrix2)
  {
    result[0]=matrix1[0]*matrix2[0]+
      matrix1[4]*matrix2[1]+
      matrix1[8]*matrix2[2]+
      matrix1[12]*matrix2[3];
    result[4]=matrix1[0]*matrix2[4]+
      matrix1[4]*matrix2[5]+
      matrix1[8]*matrix2[6]+
      matrix1[12]*matrix2[7];
    result[8]=matrix1[0]*matrix2[8]+
      matrix1[4]*matrix2[9]+
      matrix1[8]*matrix2[10]+
      matrix1[12]*matrix2[11];
    result[12]=matrix1[0]*matrix2[12]+
      matrix1[4]*matrix2[13]+
      matrix1[8]*matrix2[14]+
      matrix1[12]*matrix2[15];
    result[1]=matrix1[1]*matrix2[0]+
      matrix1[5]*matrix2[1]+
      matrix1[9]*matrix2[2]+
      matrix1[13]*matrix2[3];
    result[5]=matrix1[1]*matrix2[4]+
      matrix1[5]*matrix2[5]+
      matrix1[9]*matrix2[6]+
      matrix1[13]*matrix2[7];
    result[9]=matrix1[1]*matrix2[8]+
      matrix1[5]*matrix2[9]+
      matrix1[9]*matrix2[10]+
      matrix1[13]*matrix2[11];
    result[13]=matrix1[1]*matrix2[12]+
      matrix1[5]*matrix2[13]+
      matrix1[9]*matrix2[14]+
      matrix1[13]*matrix2[15];
    result[2]=matrix1[2]*matrix2[0]+
      matrix1[6]*matrix2[1]+
      matrix1[10]*matrix2[2]+
      matrix1[14]*matrix2[3];
    result[6]=matrix1[2]*matrix2[4]+
      matrix1[6]*matrix2[5]+
      matrix1[10]*matrix2[6]+
      matrix1[14]*matrix2[7];
    result[10]=matrix1[2]*matrix2[8]+
      matrix1[6]*matrix2[9]+
      matrix1[10]*matrix2[10]+
      matrix1[14]*matrix2[11];
    result[14]=matrix1[2]*matrix2[12]+
      matrix1[6]*matrix2[13]+
      matrix1[10]*matrix2[14]+
      matrix1[14]*matrix2[15];
    result[3]=matrix1[3]*matrix2[0]+
      matrix1[7]*matrix2[1]+
      matrix1[11]*matrix2[2]+
      matrix1[15]*matrix2[3];
    result[7]=matrix1[3]*matrix2[4]+
      matrix1[7]*matrix2[5]+
      matrix1[11]*matrix2[6]+
      matrix1[15]*matrix2[7];
    result[11]=matrix1[3]*matrix2[8]+
      matrix1[7]*matrix2[9]+
      matrix1[11]*matrix2[10]+
      matrix1[15]*matrix2[11];
    result[15]=matrix1[3]*matrix2[12]+
      matrix1[7]*matrix2[13]+
      matrix1[11]*matrix2[14]+
      matrix1[15]*matrix2[15];
  }

  void MultiplyMatrixByVector4by4OpenGL_FLOAT(float *resultvector, const float *matrix, const float *pvector)
  {
    resultvector[0]=matrix[0]*pvector[0]+matrix[4]*pvector[1]+matrix[8]*pvector[2]+matrix[12]*pvector[3];
    resultvector[1]=matrix[1]*pvector[0]+matrix[5]*pvector[1]+matrix[9]*pvector[2]+matrix[13]*pvector[3];
    resultvector[2]=matrix[2]*pvector[0]+matrix[6]*pvector[1]+matrix[10]*pvector[2]+matrix[14]*pvector[3];
    resultvector[3]=matrix[3]*pvector[0]+matrix[7]*pvector[1]+matrix[11]*pvector[2]+matrix[15]*pvector[3];
  }

#define SWAP_ROWS_DOUBLE(a, b) { double *_tmp = a; (a)=(b); (b)=_tmp; }
#define SWAP_ROWS_FLOAT(a, b) { float *_tmp = a; (a)=(b); (b)=_tmp; }
#define MAT(m,r,c) (m)[(c)*4+(r)]

  //This code comes directly from GLU except that it is for float
  int glhInvertMatrixf2(float *m, float *out)
  {
   float wtmp[4][8];
   float m0, m1, m2, m3, s;
   float *r0, *r1, *r2, *r3;
   r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];
   r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1),
      r0[2] = MAT(m, 0, 2), r0[3] = MAT(m, 0, 3),
      r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
      r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1),
      r1[2] = MAT(m, 1, 2), r1[3] = MAT(m, 1, 3),
      r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
      r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1),
      r2[2] = MAT(m, 2, 2), r2[3] = MAT(m, 2, 3),
      r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
      r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1),
      r3[2] = MAT(m, 3, 2), r3[3] = MAT(m, 3, 3),
      r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;
   /* choose pivot - or die */
   if (fabsf(r3[0]) > fabsf(r2[0]))
      SWAP_ROWS_FLOAT(r3, r2);
   if (fabsf(r2[0]) > fabsf(r1[0]))
      SWAP_ROWS_FLOAT(r2, r1);
   if (fabsf(r1[0]) > fabsf(r0[0]))
      SWAP_ROWS_FLOAT(r1, r0);
   if (0.0 == r0[0])
      return 0;
   /* eliminate first variable     */
   m1 = r1[0] / r0[0];
   m2 = r2[0] / r0[0];
   m3 = r3[0] / r0[0];
   s = r0[1];
   r1[1] -= m1 * s;
   r2[1] -= m2 * s;
   r3[1] -= m3 * s;
   s = r0[2];
   r1[2] -= m1 * s;
   r2[2] -= m2 * s;
   r3[2] -= m3 * s;
   s = r0[3];
   r1[3] -= m1 * s;
   r2[3] -= m2 * s;
   r3[3] -= m3 * s;
   s = r0[4];
   if (s != 0.0) {
      r1[4] -= m1 * s;
      r2[4] -= m2 * s;
      r3[4] -= m3 * s;
   }
   s = r0[5];
   if (s != 0.0) {
      r1[5] -= m1 * s;
      r2[5] -= m2 * s;
      r3[5] -= m3 * s;
   }
   s = r0[6];
   if (s != 0.0) {
      r1[6] -= m1 * s;
      r2[6] -= m2 * s;
      r3[6] -= m3 * s;
   }
   s = r0[7];
   if (s != 0.0) {
      r1[7] -= m1 * s;
      r2[7] -= m2 * s;
      r3[7] -= m3 * s;
   }
   /* choose pivot - or die */
   if (fabsf(r3[1]) > fabsf(r2[1]))
      SWAP_ROWS_FLOAT(r3, r2);
   if (fabsf(r2[1]) > fabsf(r1[1]))
      SWAP_ROWS_FLOAT(r2, r1);
   if (0.0 == r1[1])
      return 0;
   /* eliminate second variable */
   m2 = r2[1] / r1[1];
   m3 = r3[1] / r1[1];
   r2[2] -= m2 * r1[2];
   r3[2] -= m3 * r1[2];
   r2[3] -= m2 * r1[3];
   r3[3] -= m3 * r1[3];
   s = r1[4];
   if (0.0 != s) {
      r2[4] -= m2 * s;
      r3[4] -= m3 * s;
   }
   s = r1[5];
   if (0.0 != s) {
      r2[5] -= m2 * s;
      r3[5] -= m3 * s;
   }
   s = r1[6];
   if (0.0 != s) {
      r2[6] -= m2 * s;
      r3[6] -= m3 * s;
   }
   s = r1[7];
   if (0.0 != s) {
      r2[7] -= m2 * s;
      r3[7] -= m3 * s;
   }
   /* choose pivot - or die */
   if (fabsf(r3[2]) > fabsf(r2[2]))
      SWAP_ROWS_FLOAT(r3, r2);
   if (0.0 == r2[2])
      return 0;
   /* eliminate third variable */
   m3 = r3[2] / r2[2];
   r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
      r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];
   /* last check */
   if (0.0 == r3[3])
      return 0;
   s = 1.0 / r3[3];		/* now back substitute row 3 */
   r3[4] *= s;
   r3[5] *= s;
   r3[6] *= s;
   r3[7] *= s;
   m2 = r2[3];			/* now back substitute row 2 */
   s = 1.0 / r2[2];
   r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
      r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
   m1 = r1[3];
   r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
      r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;
   m0 = r0[3];
   r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
      r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;
   m1 = r1[2];			/* now back substitute row 1 */
   s = 1.0 / r1[1];
   r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
      r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
   m0 = r0[2];
   r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
      r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;
   m0 = r0[1];			/* now back substitute row 0 */
   s = 1.0 / r0[0];
   r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
      r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);
   MAT(out, 0, 0) = r0[4];
   MAT(out, 0, 1) = r0[5], MAT(out, 0, 2) = r0[6];
   MAT(out, 0, 3) = r0[7], MAT(out, 1, 0) = r1[4];
   MAT(out, 1, 1) = r1[5], MAT(out, 1, 2) = r1[6];
   MAT(out, 1, 3) = r1[7], MAT(out, 2, 0) = r2[4];
   MAT(out, 2, 1) = r2[5], MAT(out, 2, 2) = r2[6];
   MAT(out, 2, 3) = r2[7], MAT(out, 3, 0) = r3[4];
   MAT(out, 3, 1) = r3[5], MAT(out, 3, 2) = r3[6];
   MAT(out, 3, 3) = r3[7];
   return 1;
  }






int _glhProjectf(
                  float * position3D,
                  float *modelview,
                  float *projection,
                  int *viewport,
                  float *windowCoordinate
                )
{
      int result = 0;
      float objX=position3D[0];
      float objY=position3D[1];
      float objZ=position3D[2];
      //Transformation vectors
      float fTmp[8];
      //Modelview transform
      fTmp[0]=(modelview[0]*objX)+(modelview[4]*objY)+(modelview[8]*objZ) +modelview[12];  //w is always 1
      fTmp[1]=(modelview[1]*objX)+(modelview[5]*objY)+(modelview[9]*objZ) +modelview[13];
      fTmp[2]=(modelview[2]*objX)+(modelview[6]*objY)+(modelview[10]*objZ)+modelview[14];
      fTmp[3]=(modelview[3]*objX)+(modelview[7]*objY)+(modelview[11]*objZ)+modelview[15];
      //Projection transform, the final row of projection matrix is always [0 0 -1 0]
      //so we optimize for that.
      fTmp[4]=(projection[0]*fTmp[0])+(projection[4]*fTmp[1])+(projection[8]*fTmp[2]) +(projection[12]*fTmp[3]);
      fTmp[5]=(projection[1]*fTmp[0])+(projection[5]*fTmp[1])+(projection[9]*fTmp[2]) +(projection[13]*fTmp[3]);
      fTmp[6]=(projection[2]*fTmp[0])+(projection[6]*fTmp[1])+(projection[10]*fTmp[2])+(projection[14]*fTmp[3]);
      fTmp[7]=-fTmp[2];
      //The result normalizes between -1 and 1
      if(fTmp[7]!=0.0)	//The w value
         {
           fTmp[7]=1.0/fTmp[7];
           //Perspective division
           fTmp[4]*=fTmp[7];
           fTmp[5]*=fTmp[7];
           fTmp[6]*=fTmp[7];

           //Window coordinates
           //Map x, y to range 0-1
           windowCoordinate[0]=(fTmp[4]*0.5+0.5)*viewport[2]+viewport[0];
           windowCoordinate[1]=(fTmp[5]*0.5+0.5)*viewport[3]+viewport[1];

           //This is only correct when glDepthRange(0.0, 1.0)
           windowCoordinate[2]=(1.0+fTmp[6])*0.5;	//Between 0 and 1
           result = 1;
         }
   return result;
}

int _glhUnProjectf(float winx, float winy, float winz, float *modelview, float *projection, int *viewport, float *objectCoordinate)
{
      int result = 0;
      //Transformation matrices
      float m[16], A[16];
      float in[4], out[4];
      //Calculation for inverting a matrix, compute projection x modelview
      //and store in A[16]
      MultiplyMatrices4by4OpenGL_FLOAT(A, projection, modelview);
      //Now compute the inverse of matrix A
      if(glhInvertMatrixf2(A, m)==0) {return 0;}
      //Transformation of normalized coordinates between -1 and 1
      in[0]=(winx-(float)viewport[0])/(float)viewport[2]*2.0-1.0;
      in[1]=(winy-(float)viewport[1])/(float)viewport[3]*2.0-1.0;
      in[2]=2.0*winz-1.0;
      in[3]=1.0;

      //Objects coordinates
      MultiplyMatrixByVector4by4OpenGL_FLOAT(out, m, in);
      if(out[3]!=0.0)
         {
          out[3]=1.0/out[3];
          objectCoordinate[0]=out[0]*out[3];
          objectCoordinate[1]=out[1]*out[3];
          objectCoordinate[2]=out[2]*out[3];
          result=1;
         }

      return result;
}

int
glLookAt(
         double * m,
         double eyex,double eyey,double eyez,
         double centerx,double centery,double centerz,
         double upx,double upy, double upz
        )
{
    fprintf(stderr,"glLookAt not implemented");
    exit(1);
    /*
    double forward[3], side[3], up[3];

    forward[0] = centerx - eyex;
    forward[1] = centery - eyey;
    forward[2] = centerz - eyez;

    up[0] = upx;
    up[1] = upy;
    up[2] = upz;

    normalize(forward);

    // Side = forward x up
    cross(forward, up, side);
    normalize(side);

    // Recompute up as: up = side x forward
    cross(side, forward, up);

    __gluMakeIdentityf(&m[0][0]);




    m[0][0] = side[0];
    m[1][0] = side[1];
    m[2][0] = side[2];

    m[0][1] = up[0];
    m[1][1] = up[1];
    m[2][1] = up[2];

    m[0][2] = -forward[0];
    m[1][2] = -forward[1];
    m[2][2] = -forward[2];

    glMultMatrixf(&m[0][0]);
    glTranslated(-eyex, -eyey, -eyez);*/
}

void glGetViewportMatrix(float * m ,float startX,float startY,float width,float height ,float near ,float far)
{ //See https://en.wikibooks.org/wiki/GLSL_Programming/Vertex_Transformations
  m[0]=width/2;
  m[1]=0.0f;
  m[2]=0.0f;
  m[3]=startX + (width/2);

  m[4]=0.0f;
  m[5]=height/2;
  m[6]=0.0f;
  m[7]=startY+(height/2);

  m[8]=0.0f;
  m[9]=0.0f;
  m[10]=(far-near)/2;
  m[11]=(near+far)/2;

  m[12]=0.0f;
  m[13]=0.0f;
  m[14]=0.0f;
  m[15]=1.0f;
}



void getModelViewProjectionMatrixFromMatrices(struct Matrix4x4OfFloats * output,struct Matrix4x4OfFloats * projectionMatrix,struct Matrix4x4OfFloats * viewMatrix,struct Matrix4x4OfFloats * modelMatrix)
{
    //fprintf(stderr,"Asked To perform multiplication MVP = Projection * View * Model");

    //print4x4DMatrix("projectionMatrix",projectionMatrix,1);
    //print4x4DMatrix("viewMatrix",viewMatrix,1);
    //print4x4DMatrix("modelMatrix",modelMatrix,1);

     //MVP = Projection * View * Model || Remember, matrix multiplication is the other way around
     ///THIS IS THE CORRECT WAY TO PERFORM THE MULTIPLICATION WITH OUR ROW MAJOR MATRICES
     multiplyThree4x4FMatrices(output,projectionMatrix,viewMatrix,modelMatrix);
     //print4x4DMatrix("MVP=",output,1);
}






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
                             )
{
     int viewport[4]={0};
     buildOpenGLProjectionForIntrinsics_OpenGLColumnMajor(
                                                           projectionMatrix->m ,
                                                           viewport ,
                                                           fx, fy,
                                                           skew,
                                                           cx,  cy,
                                                           windowWidth, windowHeight,
                                                           near,
                                                           far
                                                          );
     transpose4x4FMatrix(projectionMatrix->m); //We want our own Row Major format..
     //fprintf(stderr,"viewport(%u,%u,%u,%u)\n",viewport[0],viewport[1],viewport[2],viewport[3]);
     //glViewport(viewport[0],viewport[1],viewport[2],viewport[3]); //<--Does this do anything?

     create4x4FScalingMatrix(viewMatrix,1.0,1.0,1.0);
     glGetViewportMatrix(viewportMatrix->m,viewport[0],viewport[1],viewport[2],viewport[3],near,far);
}



void correctProjectionMatrixForDifferentViewport(
                                                  float * out,
                                                  float * projectionMatrix,
                                                  float * originalViewport,
                                                  float * newViewport
                                                )
{
    //https://stackoverflow.com/questions/7604322/clip-matrix-for-3d-perspective-projection#20180585
    //https://www.opengl.org/discussion_boards/printthread.php?t=165751&page=1
    //https://github.com/gamedev-net/nehe-opengl/tree/master/linux/lesson22
    float originalViewportX = originalViewport[0];
    float originalViewportY = originalViewport[1];
    float originalViewportWidth = originalViewport[2];
    float originalViewportHeight = originalViewport[3];

    float newViewportX = newViewport[0];
    float newViewportY = newViewport[1];
    float newViewportWidth = newViewport[2];
    float newViewportHeight = newViewport[3];

	float xC =  (newViewportX - 0.5f * originalViewportWidth  - originalViewportX) / originalViewportWidth;
	float yC = -(newViewportY - 0.5f * originalViewportHeight - originalViewportY) / originalViewportHeight;
	float wC = (float) newViewportWidth/originalViewportWidth;
	float hC = (float) newViewportHeight/originalViewportHeight;

	float correction[16]={0};
	correction[0]= (1.0 / wC);
	correction[3]= -2.0 * (xC + wC / 2.0) * (1.0 / wC);
	correction[5]= (1.0 / hC);
	correction[7]= -2.0 * (yC - hC / 2.0) * (1.0 / hC);
    //transpose4x4DMatrix(correction);

    multiplyTwoRaw4x4FMatricesS(out,correction,projectionMatrix);
    //or ?
    //multiplyTwo4x4DMatrices(out,projectionMatrix,correction);
}







