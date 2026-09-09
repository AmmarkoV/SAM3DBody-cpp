#include "matrix3x3Tools.h"
#include "matrixTools.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PRINT_MATRIX_DEBUGGING 0

enum mat3x3Item
{
    I11 = 0 , I12 , I13 ,
    I21     , I22 , I23 ,
    I31     , I32 , I33
};


enum mat3x3EItem
{
    e0 = 0 , e1  , e2  ,
    e3     , e4  , e5  ,
    e6     , e7 ,  e8
};



double * alloc3x3Matrix()
{
  return malloc ( sizeof(double) * 16 );
}

void free3x3Matrix(double ** mat)
{
  if (mat==0) { return ; }
  if (*mat==0) { return ; }
  free(*mat);
  *mat=0;
}


void print3x3FMatrix(const char * str , float * matrix3x3)
{
  #if PRINT_MATRIX_DEBUGGING
  fprintf( stderr, "  3x3 float %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "%f ",matrix3x3[0]);  fprintf( stderr, "%f ",matrix3x3[1]);  fprintf( stderr, "%f\n",matrix3x3[2]);
  fprintf( stderr, "%f ",matrix3x3[3]);  fprintf( stderr, "%f ",matrix3x3[4]);  fprintf( stderr, "%f\n",matrix3x3[5]);
  fprintf( stderr, "%f ",matrix3x3[6]);  fprintf( stderr, "%f ",matrix3x3[7]);  fprintf( stderr, "%f\n",matrix3x3[8]);
  fprintf( stderr, "--------------------------------------\n");
  #endif // PRINT_MATRIX_DEBUGGING
}

void print3x3DMatrix(const char * str , double * matrix3x3)
{
  #if PRINT_MATRIX_DEBUGGING
  fprintf( stderr, "  3x3 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "%f ",matrix3x3[0]);  fprintf( stderr, "%f ",matrix3x3[1]);  fprintf( stderr, "%f\n",matrix3x3[2]);
  fprintf( stderr, "%f ",matrix3x3[3]);  fprintf( stderr, "%f ",matrix3x3[4]);  fprintf( stderr, "%f\n",matrix3x3[5]);
  fprintf( stderr, "%f ",matrix3x3[6]);  fprintf( stderr, "%f ",matrix3x3[7]);  fprintf( stderr, "%f\n",matrix3x3[8]);
  fprintf( stderr, "--------------------------------------\n");
  #endif // PRINT_MATRIX_DEBUGGING
}



void print3x3DMathematicaMatrix(const char * str , double * matrix3x3)
{
  #if PRINT_MATRIX_DEBUGGING
  fprintf( stderr, "%s = { { %f ,",str,matrix3x3[0]);  fprintf( stderr, "%f ,",matrix3x3[1]);  fprintf( stderr, "%f } , ",matrix3x3[2]);
  fprintf( stderr, "{ %f ,",matrix3x3[3]);  fprintf( stderr, "%f ,",matrix3x3[4]);  fprintf( stderr, "%f } , ",matrix3x3[5]);
  fprintf( stderr, "{ %f ,",matrix3x3[6]);  fprintf( stderr, "%f ,",matrix3x3[7]);  fprintf( stderr, "%f } }\n\n",matrix3x3[8]);
  #endif // PRINT_MATRIX_DEBUGGING
}

void print3x3DScilabMatrix(const char * str , double * matrix3x3)
{
  #if PRINT_MATRIX_DEBUGGING
  fprintf( stderr, "%s = [ %f ",str,matrix3x3[0]);  fprintf( stderr, "%f ",matrix3x3[1]);  fprintf( stderr, "%f ; ",matrix3x3[2]);
  fprintf( stderr, "%f ",matrix3x3[3]);  fprintf( stderr, "%f ",matrix3x3[4]);  fprintf( stderr, "%f ; ",matrix3x3[5]);
  fprintf( stderr, "%f ",matrix3x3[6]);  fprintf( stderr, "%f ",matrix3x3[7]);  fprintf( stderr, "%f ]\n\n",matrix3x3[8]);
  #endif // PRINT_MATRIX_DEBUGGING
}


void copy3x3Matrix(double * out,double * in)
{
  out[0]=in[0];   out[1]=in[1];   out[2]=in[2];
  out[3]=in[3];   out[4]=in[4];   out[5]=in[5];
  out[6]=in[6];   out[7]=in[7];   out[8]=in[8];
}



void create3x3IdentityMatrix(double * m)
{
    m[0] = 1.0;  m[1] = 0.0;  m[2] = 0.0;
    m[3] = 0.0;  m[4] = 1.0;  m[5] = 0.0;
    m[6] = 0.0;  m[7] = 0.0;  m[8] = 1.0;
}


void convert3x3DMatrixto3x3F(float * d, double * m )
{
    d[0]=m[0];   d[1]=m[1];   d[2]=m[2];
    d[3]=m[3];   d[4]=m[4];   d[5]=m[5];
    d[6]=m[6];   d[7]=m[7];   d[8]=m[8];
}




void  create3x3EulerVectorRotationMatrix(double * matrix3x3,double * axisXYZ,double angle)
{
 double x = axisXYZ[0];
 double y = axisXYZ[1];
 double z = axisXYZ[2];

 double radFactor = 2.0 * PI / 360.0;
 double cosA = cos(radFactor * angle);
 double sinA = sin(radFactor * angle);

  matrix3x3[0]=cosA+x*x*(1-cosA);     /*|*/    matrix3x3[1]=x*y*(1-cosA)-z*sinA;    /*|*/    matrix3x3[2]=x*z*(1-cosA)+y*sinA;
  matrix3x3[3]=y*x*(1-cosA)+z*sinA;   /*|*/    matrix3x3[4]=cosA+y*y*(1-cosA);      /*|*/    matrix3x3[5]=y*z*(1-cosA)-x*sinA;
  matrix3x3[6]=z*x*(1-cosA)-y*sinA;   /*|*/    matrix3x3[7]=z*y*(1-cosA)+x*sinA;    /*|*/    matrix3x3[8]=cosA+z*z*(1-cosA);
}



void create3x3MatrixFromEulerAnglesXYZ(double * m ,double x, double y, double z)
{
	double cr = cos( x );
	double sr = sin( x );
	double cp = cos( y );
	double sp = sin( y );
	double cy = cos( z );
	double sy = sin( z );

	m[0] = cp*cy ;
	m[1] = cp*sy;
	m[2] = -sp ;


	double srsp = sr*sp;
	double crsp = cr*sp;

	m[3] = srsp*cy-cr*sy ;
	m[4] = srsp*sy+cr*cy ;
	m[5] = sr*cp ;

	m[6] =  crsp*cy+sr*sy ;
	m[7] =  crsp*sy-sr*cy ;
	m[8]= cr*cp ;
}


void  create3x3EulerRotationXYZOrthonormalMatrix(double * matrix3x3,double * rotationsXYZ)
{
 /*
  OpenGL rotates using roll/heading/pitch , our XYZ rotation
  glTranslatef(x,y,z);
  if ( roll!=0 ) { glRotatef(roll,0.0,0.0,1.0); }
  if ( heading!=0 ) { glRotatef(heading,0.0,1.0,0.0); }
  if ( pitch!=0 ) { glRotatef(pitch,1.0,0.0,0.0); }
 */

 double firstRot[9]={0};
 double secondRot[9]={0};
 double thirdRot[9]={0};
 double tmp[9]={0};
 double axisXYZ[3]={0};

 axisXYZ[0]=0.0; axisXYZ[1]=0.0; axisXYZ[2]=1.0;
 create3x3EulerVectorRotationMatrix(firstRot,axisXYZ,rotationsXYZ[2]);

 axisXYZ[0]=0.0; axisXYZ[1]=1.0; axisXYZ[2]=0.0;
 create3x3EulerVectorRotationMatrix(secondRot,axisXYZ,rotationsXYZ[0]);

 axisXYZ[0]=1.0; axisXYZ[1]=0.0; axisXYZ[2]=0.0;
 create3x3EulerVectorRotationMatrix(thirdRot,axisXYZ,rotationsXYZ[1]);

 multiplyTwo3x3Matrices(tmp,firstRot,secondRot);
 multiplyTwo3x3Matrices(matrix3x3,tmp,thirdRot);

}




int upscale3x3Fto4x4F(float * mat4x4,float * mat3x3)
{
  if  ( (mat3x3==0)||(mat4x4==0) )   { return 0; }

  //TRANSPOSED RESULT
  mat4x4[0]=mat3x3[0]; mat4x4[1]=mat3x3[1]; mat4x4[2]=mat3x3[2];  mat4x4[3]=0.0;
  mat4x4[4]=mat3x3[3]; mat4x4[5]=mat3x3[4]; mat4x4[6]=mat3x3[5];  mat4x4[7]=0.0;
  mat4x4[8]=mat3x3[6]; mat4x4[9]=mat3x3[7]; mat4x4[10]=mat3x3[8]; mat4x4[11]=0.0;
  mat4x4[12]=0.0;      mat4x4[13]=0.0;      mat4x4[14]=0.0;       mat4x4[15]=1.0;

  return 1;
}

int transpose3x3MatrixD(double * mat)
{
  if (mat==0) { return 0; }
  /*       -------  TRANSPOSE ------->
      0   1   2             0  3  6
      3   4   5             1  4  7
      6   7   8             2  5  8   */

  double tmp;
  tmp = mat[1]; mat[1]=mat[3];  mat[3]=tmp;
  tmp = mat[2]; mat[2]=mat[6];  mat[6]=tmp;
  tmp = mat[5]; mat[5]=mat[7];  mat[7]=tmp;
  return 1;
}


int transpose3x3MatrixDFromSource(double * dest,double * source)
{
  if ( (dest==0) || (source==0) ) { return 0; }
  /*       -------  TRANSPOSE ------->
      0   1   2             0  3  6
      3   4   5             1  4  7
      6   7   8             2  5  8   */
  dest[0]=source[0]; dest[1]=source[3]; dest[2]=source[6];
  dest[3]=source[1]; dest[4]=source[4]; dest[5]=source[7];
  dest[6]=source[2]; dest[7]=source[5]; dest[8]=source[8];

  return 1;
}


int random3x3Matrix(double * mat,double minimumValues, double maximumValues)
{
 int i=0;
 unsigned int randRange=(unsigned int) maximumValues - minimumValues;

 for (i=0; i<9; i++)
 {
     mat[i]=minimumValues + rand()%randRange;
 }

 return 1;
}


double det3x3Matrix(double * mat)
{
 double * a = mat;

 double detA  = a[I11] * a[I22] * a[I33];
        detA += a[I21] * a[I32] * a[I13];
        detA += a[I31] * a[I12] * a[I23];

 //FIRST PART DONE
        detA -= a[I11] * a[I32] * a[I23];
        detA -= a[I31] * a[I22] * a[I13];
        detA -= a[I21] * a[I12] * a[I33];

 return detA;
}


int invert3x3MatrixD(double * mat,double * result)
{
 double * a = mat;
 double * b = result;
 double detA = det3x3Matrix(mat);
 if (detA==0.0)
    {
      copy3x3Matrix(result,mat);
      fprintf(stderr,"Matrix 3x3 cannot be inverted (det = 0)\n");
      return 0;
    }
 double one_div_detA = (double) 1 / detA;

 //FIRST LINE
 b[I11]  = a[I22] * a[I33] - a[I23]*a[I32]  ;
 b[I11] *= one_div_detA;

 b[I12]  = a[I13] * a[I32] - a[I12]*a[I33]  ;
 b[I12] *= one_div_detA;

 b[I13]  = a[I12] * a[I23] - a[I13]*a[I22]  ;
 b[I13] *= one_div_detA;

 //SECOND LINE
 b[I21]  = a[I23] * a[I31] - a[I21]*a[I33]  ;
 b[I21] *= one_div_detA;

 b[I22]  = a[I11] * a[I33] - a[I13]*a[I31]  ;
 b[I22] *= one_div_detA;

 b[I23]  = a[I13] * a[I21] - a[I11]*a[I23]  ;
 b[I23] *= one_div_detA;

 //THIRD LINE
 b[I31]  = a[I21] * a[I32] - a[I22]*a[I31]  ;
 b[I31] *= one_div_detA;

 b[I32]  = a[I12] * a[I31] - a[I11]*a[I32]  ;
 b[I32] *= one_div_detA;

 b[I33]  = a[I11] * a[I22] - a[I12]*a[I21]  ;
 b[I33] *= one_div_detA;

 print3x3DMatrix("Inverted Matrix From Source",a);
 print3x3DMatrix("Inverted Matrix To Target",b);

 return 1;

}


int multiplyTwo3x3Matrices(double * result , double * matrixA , double * matrixB)
{
  if ( (matrixA==0) || (matrixB==0) || (result==0) ) { return 0; }


  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"Multiplying 3x3 A and B \n");
   print3x3DMatrix("A", matrixA);
   print3x3DMatrix("B", matrixB);
  #endif

  //MULTIPLICATION_RESULT FIRST ROW
  // 0 1 2
  // 3 4 5
  // 6 7 8
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[3]  + matrixA[2] * matrixB[6];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[7];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[8];

  result[3]=matrixA[3] * matrixB[0] + matrixA[4] * matrixB[3]  + matrixA[5] * matrixB[6];
  result[4]=matrixA[3] * matrixB[1] + matrixA[4] * matrixB[4]  + matrixA[5] * matrixB[7];
  result[5]=matrixA[3] * matrixB[2] + matrixA[4] * matrixB[5]  + matrixA[5] * matrixB[8];

  result[6]=matrixA[6] * matrixB[0] + matrixA[7] * matrixB[3]  + matrixA[8] * matrixB[6];
  result[7]=matrixA[6] * matrixB[1] + matrixA[7] * matrixB[4]  + matrixA[8] * matrixB[7];
  result[8]=matrixA[6] * matrixB[2] + matrixA[7] * matrixB[5]  + matrixA[8] * matrixB[8];

  print3x3DMatrix("AxB", result);

  return 1;
}


int transform2DPointVectorUsing3x3Matrix(double * resultPoint2D, double * transformation3x3, double * point2D)
{
  if ( (resultPoint2D==0) || (transformation3x3==0) || (point2D==0) ) { return 0; }

/*
  fprintf(stderr,"Point 2D %0.2f,%0.2f \n",point2D[0],point2D[1]);
  fprintf(stderr,"Getting multiplied with \n");
  print3x3DMatrix("transformation3x3", transformation3x3);
*/

/*
   What we want to do ( in mathematica )
   { {e0,e1,e2} , {e3,e4,e5} , {e6,e7,e8} } * { { X } , { Y } , { W } }

   This gives us

  {
    {e2 W + e0 X + e1 Y},
    {e5 W + e3 X + e4 Y},
    {e8 W + e6 X + e7 Y}
  }
*/

  double * m = transformation3x3;
  double X=point2D[0],Y=point2D[1],W=point2D[2];


  resultPoint2D[0] =  m[e2] * W + m[e0] * X + m[e1] * Y;
  resultPoint2D[1] =  m[e5] * W + m[e3] * X + m[e4] * Y;
  resultPoint2D[2] =  m[e8] * W + m[e6] * X + m[e7] * Y;

  // Ok we have our results but now to normalize our vector
  if(resultPoint2D[2]!=0.0)
  {
   resultPoint2D[0]/=resultPoint2D[2];
   resultPoint2D[1]/=resultPoint2D[2];
   resultPoint2D[2]=1.0; //resultPoint2D[2]/=resultPoint2D[2];
  } else
  {
     fprintf(stderr,"Error with W coordinate after multiplication of 2D Point with 3x3 Matrix\n");
  }
 return 1;
}


int normalize2DPointVector(double * vec)
{
  if ( vec[2]==0.0 )
  {
    fprintf(stderr,"normalize2DPointVector cannot be normalized since element 2 is zero\n");
    return 0;
  }
  if ( vec[2]==1.0 ) { return 1; }

  vec[0]=vec[0]/vec[2];
  vec[1]=vec[1]/vec[2];
  vec[2]=1.0; //vec[2]/vec[2];

  return 1;
}
