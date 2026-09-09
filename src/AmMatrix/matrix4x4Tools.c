
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix4x4Tools.h"

#define OPTIMIZED 1

#ifndef INTEL_OPTIMIZATIONS
#define INTEL_OPTIMIZATIONS 0
#endif

// AVX2 + FMA code path.  Requires GCC/Clang flag: -mavx2 -mfma
// Override via -DINTEL_AVX2_OPTIMIZATIONS=1 at build time (see makeLibrary.sh).
#ifndef INTEL_AVX2_OPTIMIZATIONS
#define INTEL_AVX2_OPTIMIZATIONS 0
#endif


#if INTEL_OPTIMIZATIONS
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#if INTEL_AVX2_OPTIMIZATIONS
#include <immintrin.h>   /* AVX2 + FMA3 */
#endif

int codeHasSSE()
{
#if INTEL_OPTIMIZATIONS
 return 2; //2=>SSE2
#else
 return 0;
#endif // INTEL_OPTIMIZATIONS
}

int codeHasAVX2()
{
#if INTEL_AVX2_OPTIMIZATIONS
 return 1;
#else
 return 0;
#endif // INTEL_AVX2_OPTIMIZATIONS
}

#define PRINT_MATRIX_DEBUGGING 0

enum mat4x4Item
{
    I11 = 0 , I12 , I13 , I14 ,
    I21     , I22 , I23 , I24 ,
    I31     , I32 , I33 , I34 ,
    I41     , I42 , I43 , I44
};

enum mat4x4RTItem
{
    r0 = 0 , r1, r2 , t0 ,
    r3     , r4, r5 , t1 ,
    r6     , r7 ,r8 , t2 ,
    zero0  , zero1 , zero2 , one0
};

enum mat4x4EItem
{
    e0 = 0 , e1  , e2  , e3 ,
    e4     , e5  , e6  , e7 ,
    e8     , e9  , e10 , e11 ,
    e12    , e13 , e14 , e15
};

/* OUR MATRICES STORAGE
    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15
*/

const float __attribute__((aligned(16))) identityMatrix4x4[16]={1.0,0.0,0.0,0.0,
                                                                0.0,1.0,0.0,0.0,
                                                                0.0,0.0,1.0,0.0,
                                                                0.0,0.0,0.0,1.0};

void print4x4FMatrix(const char * str , float * matrix4x4,int forcePrint)
{
  #if PRINT_MATRIX_DEBUGGING
   forcePrint=1;
  #endif // PRINT_MATRIX_DEBUGGING

  if (forcePrint)
  {
  fprintf( stderr, " 4x4 float %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[0],matrix4x4[1],matrix4x4[2],matrix4x4[3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[4],matrix4x4[5],matrix4x4[6],matrix4x4[7]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[8],matrix4x4[9],matrix4x4[10],matrix4x4[11]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f\n",matrix4x4[12],matrix4x4[13],matrix4x4[14],matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");
  }
}

void print4x4DMatrix(const char * str , double * matrix4x4,int forcePrint)
{
  #if PRINT_MATRIX_DEBUGGING
   forcePrint=1;
  #endif // PRINT_MATRIX_DEBUGGING

  if (forcePrint)
  {
  fprintf( stderr, " 4x4 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[0],matrix4x4[1],matrix4x4[2],matrix4x4[3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[4],matrix4x4[5],matrix4x4[6],matrix4x4[7]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[8],matrix4x4[9],matrix4x4[10],matrix4x4[11]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f\n",matrix4x4[12],matrix4x4[13],matrix4x4[14],matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");
  }
}


void print4x4FMathematicaMatrix(const char * str , float * matrix3x3,int forcePrint)
{
  #if PRINT_MATRIX_DEBUGGING
    forcePrint=1;
  #endif // PRINT_MATRIX_DEBUGGING
  if (forcePrint)
  {
  fprintf( stderr, "%s = { { %f , %f , %f ,%f } , { %f , %f , %f , %f } , { %f , %f , %f , %f } , { %f , %f , %f , %f } }\n",str,
           matrix3x3[0],matrix3x3[1],matrix3x3[2],matrix3x3[3],
           matrix3x3[4],matrix3x3[5],matrix3x3[6],matrix3x3[7],
           matrix3x3[8],matrix3x3[9],matrix3x3[10],matrix3x3[11],
           matrix3x3[12],matrix3x3[13],matrix3x3[14],matrix3x3[15]);
  }
}



void copy3x3FMatrixTo4x4F(float * out,float * in)
{
  if ((out!=0) && (in!=0))
  {
   out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=0.0;
   out[4]=in[3];   out[5]=in[4];   out[6]=in[5];   out[7]=0.0;
   out[8]=in[6];   out[9]=in[7];   out[10]=in[8];  out[11]=0.0;
   out[12]=0.0;    out[13]=0.0;    out[14]=0.0;    out[15]=1.0;
  }
}



void copy4x4FMatrix(float * out,const float * in)
{
  if ((out!=0) && (in!=0))
  {
   #if INTEL_AVX2_OPTIMIZATIONS
    /* Two 256-bit stores cover all 16 floats in two cache-line halves */
    _mm256_storeu_ps(&out[0], _mm256_loadu_ps(&in[0]));
    _mm256_storeu_ps(&out[8], _mm256_loadu_ps(&in[8]));
   #elif OPTIMIZED
    memcpy(out,in,16*sizeof(float));
   #else
    out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=in[3];
    out[4]=in[4];   out[5]=in[5];   out[6]=in[6];   out[7]=in[7];
    out[8]=in[8];   out[9]=in[9];   out[10]=in[10]; out[11]=in[11];
    out[12]=in[12]; out[13]=in[13]; out[14]=in[14]; out[15]=in[15];
   #endif // OPTIMIZED
  }
}


void copy4x4FMatrixToAlignedContainer(struct Matrix4x4OfFloats * out,const float * in)
{
  if (out!=0)
    { copy4x4FMatrix(out->m,in); }
}

void copy4x4DMatrix(double * out,double * in)
{
  if ((out!=0) && (in!=0))
  {
   #if OPTIMIZED
    memcpy(out,in,16*sizeof(double));
   #else
    out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=in[3];
    out[4]=in[4];   out[5]=in[5];   out[6]=in[6];   out[7]=in[7];
    out[8]=in[8];   out[9]=in[9];   out[10]=in[10]; out[11]=in[11];
    out[12]=in[12]; out[13]=in[13]; out[14]=in[14]; out[15]=in[15];
   #endif // OPTIMIZED
  }
}


void copy4x4FMatrixTo4x4D(double * out,float * in)
{
  out[0]=(double)  in[0];   out[1]= (double) in[1];   out[2]= (double) in[2];  out[3]= (double) in[3];
  out[4]=(double)  in[4];   out[5]= (double) in[5];   out[6]= (double) in[6];  out[7]= (double) in[7];
  out[8]=(double)  in[8];   out[9]= (double) in[9];   out[10]=(double) in[10]; out[11]=(double) in[11];
  out[12]=(double) in[12];  out[13]=(double) in[13];  out[14]=(double) in[14]; out[15]=(double) in[15];
}

void copy4x4DMatrixTo4x4F(float * dest, double * m )
{
    dest[0]=(float)m[0];   dest[1]=(float)m[1];   dest[2]=(float)m[2];    dest[3]=(float)m[3];
    dest[4]=(float)m[4];   dest[5]=(float)m[5];   dest[6]=(float)m[6];    dest[7]=(float)m[7];
    dest[8]=(float)m[8];   dest[9]=(float)m[9];   dest[10]=(float)m[10];  dest[11]=(float)m[11];
    dest[12]=(float)m[12]; dest[13]=(float)m[13]; dest[14]=(float)m[14];  dest[15]=(float)m[15];
}


void create4x4FIdentityMatrixDirect(float * m)
{
    if (m!=0)
    {
     memcpy(m,identityMatrix4x4,16*sizeof(float));
     /*
     memset(m,0,16*sizeof(float));
     m[0] = 1.0;
     m[5] = 1.0;
     m[10] = 1.0;
     m[15] = 1.0;*/
    }
}

void create4x4FIdentityMatrix(struct Matrix4x4OfFloats * m)
{
   /*#if INTEL_OPTIMIZATIONS
    // 0.79 instruction fetch
    __m128 zero = _mm_setzero_ps();
    _mm_store_ps(&m->m[0], zero);
    m->m[0] = 1.0;
    _mm_store_ps(&m->m[4], zero);
    m->m[5] = 1.0;
    _mm_store_ps(&m->m[8], zero);
    m->m[10] = 1.0;
    _mm_store_ps(&m->m[12], zero);
    m->m[15] = 1.0;
    return;
   #else
   */

   //0.60
   memcpy(m->m,identityMatrix4x4,16*sizeof(float));

/*
   #if OPTIMIZED
    // 0.61 instruction fetch
    memset(m->m,0,16*sizeof(float));
    m->m[0] = 1.0;
    m->m[5] = 1.0;
    m->m[10] = 1.0;
    m->m[15] = 1.0;
   return;
  #else
    //0.77 instruction fetch
    m->m[0] = 1.0;  m->m[1] = 0.0;  m->m[2] = 0.0;   m->m[3] = 0.0;
    m->m[4] = 0.0;  m->m[5] = 1.0;  m->m[6] = 0.0;   m->m[7] = 0.0;
    m->m[8] = 0.0;  m->m[9] = 0.0;  m->m[10] = 1.0;  m->m[11] =0.0;
    m->m[12]= 0.0;  m->m[13]= 0.0;  m->m[14] = 0.0;  m->m[15] = 1.0;
    return;
  #endif // OPTIMIZED
  */
 // #endif //INTEL optimizations are more optimizing.. :P
}



//static inline
float degrees_to_radF(float degrees)
{
 return (float) degrees * ((float) M_PI / 180.0);
}




int floatPEq(float * element , float value )
{
 const float machineFloatPrecision= 0.0001; // 0.0001; original
 if ( *element == value ) { return 1; }

 if ( ( value-machineFloatPrecision<*element ) && (( *element<value+machineFloatPrecision )) ) { return 1; }
 return 0;
}


int is4x4FIdentityMatrix(float  * m)
{
   return (
           (floatPEq(&m[0],1.0)) &&(floatPEq(&m[1],0.0)) &&(floatPEq(&m[2],0.0)) &&(floatPEq(&m[3],0.0)) &&
           (floatPEq(&m[4],0.0)) &&(floatPEq(&m[5],1.0)) &&(floatPEq(&m[6],0.0)) &&(floatPEq(&m[7],0.0)) &&
           (floatPEq(&m[8],0.0)) &&(floatPEq(&m[9],0.0)) &&(floatPEq(&m[10],1.0))&&(floatPEq(&m[11],0.0))&&
           (floatPEq(&m[12],0.0))&&(floatPEq(&m[13],0.0))&&(floatPEq(&m[14],0.0))&&(floatPEq(&m[15],1.0))
          );
}


int is4x4FZeroMatrix(float  * m)
{
   return (
            (floatPEq(&m[0] ,0.0))&&(floatPEq(&m[1], 0.0))&&(floatPEq(&m[2] ,0.0))&&(floatPEq(&m[3] ,0.0))&&
            (floatPEq(&m[4] ,0.0))&&(floatPEq(&m[5], 0.0))&&(floatPEq(&m[6] ,0.0))&&(floatPEq(&m[7] ,0.0))&&
            (floatPEq(&m[8] ,0.0))&&(floatPEq(&m[9], 0.0))&&(floatPEq(&m[10],0.0))&&(floatPEq(&m[11],0.0))&&
            (floatPEq(&m[12],0.0))&&(floatPEq(&m[13],0.0))&&(floatPEq(&m[14],0.0))&&(floatPEq(&m[15],0.0))
           );
}


int are4x4FMatricesEqual(float * m1,float * m2)
{
   return (
           (floatPEq(&m1[0],m2[0])) &&(floatPEq(&m1[1],m2[1])) &&(floatPEq(&m1[2],m2[2])) &&(floatPEq(&m1[3],m2[3])) &&
           (floatPEq(&m1[4],m2[4])) &&(floatPEq(&m1[5],m2[5])) &&(floatPEq(&m1[6],m2[6])) &&(floatPEq(&m1[7],m2[7])) &&
           (floatPEq(&m1[8],m2[8])) &&(floatPEq(&m1[9],m2[9])) &&(floatPEq(&m1[10],m2[10]))&&(floatPEq(&m1[11],m2[11]))&&
           (floatPEq(&m1[12],m2[12]))&&(floatPEq(&m1[13],m2[13]))&&(floatPEq(&m1[14],m2[14]))&&(floatPEq(&m1[15],m2[15]))
          );
}



int is4x4FIdentityMatrixS(struct Matrix4x4OfFloats * m)
{
   return is4x4FIdentityMatrix(m->m);
}


int is4x4FIdentityMatrixPercisionCompensating(struct Matrix4x4OfFloats  * m)
{
   return (
              (floatPEq(&m->m[0],1.0)) &&(floatPEq(&m->m[1],0.0)) &&(floatPEq(&m->m[2],0.0)) &&(floatPEq(&m->m[3],0.0)) &&
              (floatPEq(&m->m[4],0.0)) &&(floatPEq(&m->m[5],1.0)) &&(floatPEq(&m->m[6],0.0)) &&(floatPEq(&m->m[7],0.0)) &&
              (floatPEq(&m->m[8],0.0)) &&(floatPEq(&m->m[9],0.0)) &&(floatPEq(&m->m[10],1.0))&&(floatPEq(&m->m[11],0.0))&&
              (floatPEq(&m->m[12],0.0))&&(floatPEq(&m->m[13],0.0))&&(floatPEq(&m->m[14],0.0))&&(floatPEq(&m->m[15],1.0))
           );
}


void convert4x4FMatrixToRPY(struct Matrix4x4OfFloats * m ,float *roll,float *pitch,float *yaw)
{
 if (m->m[0] == 1.0f)
        {
          *yaw = atan2f( m->m[2], m->m[11]);
          *pitch = 0;
          *roll = 0;
        }
        else
 if (m->m[0] == -1.0f)
        {
          *yaw = atan2f(m->m[2], m->m[11]);
          *pitch = 0;
          *roll = 0;
        }
        else
        {
          *yaw = atan2(-m->m[8],m->m[0]);
          *pitch = asin(m->m[4]);
          *roll = atan2(-m->m[6],m->m[5]);
        }
}



int create4x4FMatrixFromRodriguez(struct Matrix4x4OfFloats * m,float rodriguezX,float rodriguezY,float rodriguezZ)
{
  if (m==0) { return 0; }

  float x = rodriguezX  * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
  float y = rodriguezY  * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
  float z = rodriguezZ  * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
  //-----------------------------------
  float th = sqrt( x*x + y*y + z*z );
  float cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  if ( th < 0.00001 )
    {
       create4x4FIdentityMatrix(m);
       return 1;
    }

   //NORMAL RESULT
   memcpy(m->m,identityMatrix4x4,16*sizeof(float));
   //-------------------------------------------------------------------------------------------
   m->m[0]=x*x * (1 - cosTh) + cosTh;          m->m[1]=x*y*(1 - cosTh) - z*sin(th);      m->m[2]=x*z*(1 - cosTh) + y*sin(th);   //m->m[3]=0.0;
   m->m[4]=x*y*(1 - cosTh) + z*sin(th);        m->m[5]=y*y*(1 - cosTh) + cosTh;          m->m[6]=y*z*(1 - cosTh) - x*sin(th);   //m->m[7]=0.0;
   m->m[8]=x*z*(1 - cosTh) - y*sin(th);        m->m[9]=y*z*(1 - cosTh) + x*sin(th);      m->m[10]=z*z*(1 - cosTh) + cosTh;      //m->m[11]=0.0;
   //m->m[12]=0.0;                               m->m[13]=0.0;                             m->m[14]=0.0;                          m->m[15]=1.0;

  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"rodriguez %f %f %f\n ",rodriguezX,rodriguezY,rodriguezZ);
   print4x4FMatrix("Rodriguez Initial", result);
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}



void create4x4FMatrixFromEulerAnglesXYZAllInOne(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ)
{   // https://github.com/AmmarkoV/RGBDAcquisition/blob/master/tools/AmMatrix/rotationMatrixGeneration.m
    float x = (float) eulX * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulX);
    float y = (float) eulY * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulY);
    float z = (float) eulZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulZ);

    float cr = cos( x );
    float sr = sin( x );
    float cp = cos( y );
    float sp = sin( y );
    float cy = cos( z );
    float sy = sin( z );
    float srsp = sr*sp;
    float crsp = cr*sp;

    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //-------------------------------------------------------------------------------------------
    m->m[0] = cp*cy;          m->m[1] = cp*sy;           m->m[2] = -sp;    //m->m[3] = 0;
    m->m[4] = srsp*cy-cr*sy;  m->m[5] = srsp*sy+cr*cy;   m->m[6] = sr*cp;  //m->m[7] = 0;
    m->m[8] =  crsp*cy+sr*sy; m->m[9] =  crsp*sy-sr*cy;  m->m[10]= cr*cp;  //m->m[11]= 0;
    //m->m[12]= 0;              m->m[13]= 0;               m->m[14]= 0;      m->m[15]= 1.0;
}


void create4x4FMatrixFromEulerAnglesZYX(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ)
{   // https://github.com/AmmarkoV/RGBDAcquisition/blob/master/tools/AmMatrix/rotationMatrixGeneration.m
    //roll = X , pitch = Y , yaw = Z
    float x = (float) eulX * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulX);
    float y = (float) eulY * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulY);
    float z = (float) eulZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulZ);

    float cr = cos(z);
    float sr = sin(z);
    float cp = cos(y);
    float sp = sin(y);
    float cy = cos(x);
    float sy = sin(x);

    float srsp = sr*sp;
    float crsp = cr*sp;

    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //-------------------------------------------------------------------------------------------
    m->m[0]  = cr*cp;  m->m[1]  = crsp*sy - sr*cy;   m->m[2]  = crsp*cy + sr*sy;  //m->m[3]  = 0.0;
    m->m[4]  = sr*cp;  m->m[5]  = srsp*sy + cr*cy;   m->m[6]  = srsp*cy - cr*sy;  //m->m[7]  = 0.0;
    m->m[8]  = -sp;    m->m[9]  = cp*sy;             m->m[10] = cp*cy;            //m->m[11] = 0.0;
    //m->m[12] = 0;      m->m[13] = 0;                 m->m[14] = 0;                m->m[15] = 1.0;
}


void create4x4FMatrixFromEulerAnglesZXY(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ)
{   // https://github.com/AmmarkoV/RGBDAcquisition/blob/master/tools/AmMatrix/rotationMatrixGeneration.m
    //roll = X , pitch = Y , yaw = Z
    float x = (float) eulX * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulX);
    float y = (float) eulY * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulY);
    float z = (float) eulZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(eulZ);

    float cosZ = cos(z);
    float sinZ = sin(z);
    float cosY = cos(y);
    float sinY = sin(y);
    float cosX = cos(x);
    float sinX = sin(x);

    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //-------------------------------------------------------------------------------------------
    m->m[0]  = cosY*cosZ + sinX*sinY*sinZ;    m->m[1]  = cosX * sinZ;    m->m[2]  = cosY*sinX*sinZ - cosZ*sinY;  //m->m[3]  = 0.0;
    m->m[4]  = -cosY*sinZ + cosZ*sinX*sinY;   m->m[5]  = cosX * cosZ;    m->m[6]  = cosY*cosZ*sinX + sinY*sinZ;  //m->m[7]  = 0.0;
    m->m[8]  = cosX*sinY;                     m->m[9]  = -sinX;          m->m[10] = cosX*cosY;                   //m->m[11] = 0.0;
    //m->m[12] = 0;                             m->m[13] = 0;              m->m[14] = 0;                           m->m[15] = 1.0;
}


//---------------------------------------------------------
void create4x4FRotationX(struct Matrix4x4OfFloats * m,float degrees)
{
    if (degrees!=0.0)
    {
    float radians = (float) degrees * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX    = (float) cosf((float)radians);
    float sinX    = (float) sinf((float)radians);


    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //-------------------------------------------------------------------------------------------
    //m->m[0] = 1.0;  m->m[1] = 0.0;     m->m[2]  = 0.0;    m->m[3]  = 0.0;
    //m->m[4] = 0.0;
    m->m[5] = cosX;    m->m[6]  = sinX;   //m->m[7]  = 0.0;
    //m->m[8] = 0.0;
    m->m[9] = -1*sinX; m->m[10] = cosX;   //m->m[11] = 0.0;
    //m->m[12]= 0.0;  m->m[13]= 0.0;     m->m[14] = 0.0;    m->m[15] = 1.0;
    } else
    {
      create4x4FIdentityMatrix(m);
    }
}
//---------------------------------------------------------
void create4x4FRotationY(struct Matrix4x4OfFloats * m,float degrees)
{
    if (degrees!=0.0)
    {
     float radians = (float) degrees * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
     float cosY = (float) cosf((float)radians);
     float sinY = (float) sinf((float)radians);

     m->m[0] = cosY;  m->m[1] = 0.0;  m->m[2]  = -1*sinY; m->m[3] = 0.0;
     m->m[4] = 0.0;   m->m[5] = 1.0;  m->m[6]  = 0.0;     m->m[7] = 0.0;
     m->m[8] = sinY;  m->m[9] = 0.0;  m->m[10] = cosY;    m->m[11] =0.0;
     m->m[12]= 0.0;   m->m[13]= 0.0;  m->m[14] = 0.0;     m->m[15] = 1.0;
    }
     else
    {
      create4x4FIdentityMatrix(m);
    }
    // Rotate Y formula.
    //create4x4FIdentityMatrix(m);
    //m->m[0] =    cosV; // [0,0]
    //m->m[2] = -1*sinV; // [2,0]
    //m->m[8] =    sinV; // [0,2]
    //m->m[10] =   cosV; // [2,2]
}
//---------------------------------------------------------
void create4x4FRotationZ(struct Matrix4x4OfFloats * m,float degrees)
{
    if (degrees!=0.0)
    {
    float radians = (float) degrees * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radians);
    float sinZ = (float) sinf((float)radians);

    m->m[0] = cosZ;    m->m[1] = sinZ;  m->m[2]  = 0.0; m->m[3]  = 0.0;
    m->m[4] = -1*sinZ; m->m[5] = cosZ;  m->m[6]  = 0.0; m->m[7]  = 0.0;
    m->m[8] = 0.0;     m->m[9] = 0.0;   m->m[10] = 1.0; m->m[11] = 0.0;
    m->m[12]= 0.0;     m->m[13]= 0.0;   m->m[14] = 0.0; m->m[15] = 1.0;
    }
     else
    {
      create4x4FIdentityMatrix(m);
    }

    // Rotate Z formula.
    //create4x4FIdentityMatrix(m);
    //m->m[0] =    cosV;  // [0,0]
    //m->m[1] =    sinV;  // [1,0]
    //m->m[4] = -1*sinV;  // [0,1]
    //m->m[5] =    cosV;  // [1,1]
}
//---------------------------------------------------------

void create4x4FRotationXYZ(struct Matrix4x4OfFloats * m,float degreesX,float degreesY,float degreesZ)
{
    //---------------------------------------------------------
    float radiansX = (float) degreesX * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX = (float) cosf((float)radiansX);
    float sinX = (float) sinf((float)radiansX);
    //---------------------------------------------------------
    float radiansY = (float) degreesY * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosY = (float) cosf((float)radiansY);
    float sinY = (float) sinf((float)radiansY);
    //---------------------------------------------------------
    float radiansZ = (float) degreesZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radiansZ);
    float sinZ = (float) sinf((float)radiansZ);
    //---------------------------------------------------------
    //                       == RX ==                                        == RY ==                                                  == RZ ==
    //{ {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} } * { { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } } * { { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  }
    //https://www.wolframalpha.com/input?i=%7B+%7B1%2C+0+%2C0%7D+%2C+%7B0%2C+cosX%2C+sinX%7D%2C+%7B0%2C+-sinX%2C+cosX%7D+%7D+*+%7B+%7B+cosY%2C+0%2C+-sinY+%7D%2C+%7B0%2C+1+%2C0+%7D%2C+%7B+sinY%2C+0+%2C+cosY+%7D+%7D+*+%7B+%7B+cosZ%2C+sinZ%2C+0+%7D+%2C+%7B+-sinZ%2C+cosZ+%2C+0%7D+%2C+%7B0%2C0%2C1%7D++%7D++
    //--------------
    //    Row 1
    //--------------
    m->m[0] = cosY * cosZ;
    m->m[1] = cosY * sinZ;
    m->m[2] = -sinY;
    m->m[3] = 0.0;
    //--------------
    //    Row 2
    //--------------
    m->m[4] = (sinX * sinY * cosZ) - (cosX*sinZ);
    m->m[5] = (sinX * sinY * sinZ) + (cosX*cosZ);
    m->m[6] = sinX * cosY;
    m->m[7] = 0.0;
    //--------------
    //    Row 3
    //--------------
    m->m[8]  = (cosX * sinY * cosZ) + (sinX * sinZ);
    m->m[9]  = (cosX * sinY * sinZ) - (sinX * cosZ);
    m->m[10] = cosX * cosY;
    m->m[11] = 0.0;
    //--------------
    //    Row 4
    //--------------
    m->m[12] = 0.0;
    m->m[13] = 0.0;
    m->m[14] = 0.0;
    m->m[15] = 1.0;
    //--------------
    return;
}

void create4x4FRotationXZY(struct Matrix4x4OfFloats * m,float degreesX,float degreesY,float degreesZ)
{
    //---------------------------------------------------------
    float radiansX = (float) degreesX * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX = (float) cosf((float)radiansX);
    float sinX = (float) sinf((float)radiansX);
    //---------------------------------------------------------
    float radiansY = (float) degreesY * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosY = (float) cosf((float)radiansY);
    float sinY = (float) sinf((float)radiansY);
    //---------------------------------------------------------
    float radiansZ = (float) degreesZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radiansZ);
    float sinZ = (float) sinf((float)radiansZ);
    //---------------------------------------------------------
    //                       == RX ==                                        == RZ ==                                                  == RY ==
    // { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} } * { { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  } *  { { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }
    //https://www.wolframalpha.com/input?i=%7B+cosY%2C+0%2C+-sinY+%7D%2C+%7B0%2C+1+%2C0+%7D%2C+%7B+sinY%2C+0+%2C+cosY+%7D+%7D+*+%7B+%7B+cosZ%2C+sinZ%2C+0+%7D+%2C+%7B+-sinZ%2C+cosZ+%2C+0%7D+%2C+%7B0%2C0%2C1%7D++%7D+*+%7B+%7B1%2C+0+%2C0%7D+%2C+%7B0%2C+cosX%2C+sinX%7D%2C+%7B0%2C+-sinX%2C+cosX%7D+%7D+
    //--------------
    //    Row 1
    //--------------
    m->m[0] = cosY * cosZ;
    m->m[1] = sinZ;
    m->m[2] = sinY * (-cosZ);
    m->m[3] = 0.0;
    //--------------
    //    Row 2
    //--------------
    m->m[4] = (sinX * sinY) - (cosX * cosY * sinZ);
    m->m[5] = cosX * cosZ;
    m->m[6] = (cosX * sinY * sinZ) + (sinX * cosY);
    m->m[7] = 0.0;
    //--------------
    //    Row 3
    //--------------
    m->m[8]  = (sinX * cosY * sinZ) + (cosX * sinY);
    m->m[9]  = sinX * (-cosZ);
    m->m[10] = (cosX * cosY) - (sinX * sinY * sinZ);
    m->m[11] = 0.0;
    //--------------
    //    Row 4
    //--------------
    m->m[12] = 0.0;
    m->m[13] = 0.0;
    m->m[14] = 0.0;
    m->m[15] = 1.0;
    //--------------
    return;
}




void create4x4FRotationYXZ(struct Matrix4x4OfFloats * m,float degreesX,float degreesY,float degreesZ)
{
    //---------------------------------------------------------
    float radiansX = (float) degreesX * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX = (float) cosf((float)radiansX);
    float sinX = (float) sinf((float)radiansX);
    //---------------------------------------------------------
    float radiansY = (float) degreesY * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosY = (float) cosf((float)radiansY);
    float sinY = (float) sinf((float)radiansY);
    //---------------------------------------------------------
    float radiansZ = (float) degreesZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radiansZ);
    float sinZ = (float) sinf((float)radiansZ);
    //---------------------------------------------------------
    //                       == RY ==                                        == RX ==                                                  == RZ ==
    //{ { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } } * { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} } * { { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  }
    //https://www.wolframalpha.com/input?i=%7B+%7B+cosY%2C+0%2C+-sinY+%7D%2C+%7B0%2C+1+%2C0+%7D%2C+%7B+sinY%2C+0+%2C+cosY+%7D+%7D+*+%7B+%7B1%2C+0+%2C0%7D+%2C+%7B0%2C+cosX%2C+sinX%7D%2C+%7B0%2C+-sinX%2C+cosX%7D+%7D+*+%7B+%7B+cosZ%2C+sinZ%2C+0+%7D+%2C+%7B+-sinZ%2C+cosZ+%2C+0%7D+%2C+%7B0%2C0%2C1%7D++%7D
    //--------------
    //    Row 1
    //--------------
    m->m[0] = (cosY*cosZ) - (sinX * sinY * sinZ);
    m->m[1] = (sinX * sinY * cosZ) + (cosY * sinZ);
    m->m[2] = -cosX * sinY;
    m->m[3] = 0.0;
    //--------------
    //    Row 2
    //--------------
    m->m[4] = -cosX * sinZ;
    m->m[5] = cosX * cosZ;
    m->m[6] = sinX;
    m->m[7] = 0.0;
    //--------------
    //    Row 3
    //--------------
    m->m[8]  = (sinX * cosY * sinZ) + (sinY * cosZ);
    m->m[9]  = (sinY * sinZ) - (sinX * cosY * cosZ);
    m->m[10] = cosX * cosY;
    m->m[11] = 0.0;
    //--------------
    //    Row 4
    //--------------
    m->m[12] = 0.0;
    m->m[13] = 0.0;
    m->m[14] = 0.0;
    m->m[15] = 1.0;
    //--------------
    return;
}





void create4x4FRotationYZX(struct Matrix4x4OfFloats * m,float degreesX,float degreesY,float degreesZ)
{
    //---------------------------------------------------------
    float radiansX = (float) degreesX * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX = (float) cosf((float)radiansX);
    float sinX = (float) sinf((float)radiansX);
    //---------------------------------------------------------
    float radiansY = (float) degreesY * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosY = (float) cosf((float)radiansY);
    float sinY = (float) sinf((float)radiansY);
    //---------------------------------------------------------
    float radiansZ = (float) degreesZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radiansZ);
    float sinZ = (float) sinf((float)radiansZ);
    //---------------------------------------------------------
    //                       == RY ==                                        == RZ ==                                                  == RX ==
    //{ { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } } * { { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  } * { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} }
    //https://www.wolframalpha.com/input?i=%7B+cosY%2C+0%2C+-sinY+%7D%2C+%7B0%2C+1+%2C0+%7D%2C+%7B+sinY%2C+0+%2C+cosY+%7D+%7D+*+%7B+%7B+cosZ%2C+sinZ%2C+0+%7D+%2C+%7B+-sinZ%2C+cosZ+%2C+0%7D+%2C+%7B0%2C0%2C1%7D++%7D+*+%7B+%7B1%2C+0+%2C0%7D+%2C+%7B0%2C+cosX%2C+sinX%7D%2C+%7B0%2C+-sinX%2C+cosX%7D+%7D+
    //--------------
    //    Row 1
    //--------------
    //{{Cos[Y], 0, -Sin[Y]}, {0, 1, 0}, {Cos[Z] Sin[Y], -(Cos[Y] Sin[X]) + Cos[X] Sin[Y] Sin[Z], Cos[X] Cos[Y] + Sin[X] Sin[Y] Sin[Z]}}
    m->m[0] = cosY;
    m->m[1] = 0;
    m->m[2] = -sinY;
    m->m[3] = 0.0;
    //--------------
    //    Row 2
    //--------------
    m->m[4] = 0.0;
    m->m[5] = 1.0;
    m->m[6] = 0.0;
    m->m[7] = 0.0;
    //--------------
    //    Row 3
    //--------------
    m->m[8]  = sinY * cosZ;
    m->m[9]  = (cosX * sinY * sinZ) - (sinX * cosY);
    m->m[10] = (sinX*sinY*sinZ) + (cosX * cosY);
    m->m[11] = 0.0;
    //--------------
    //    Row 4
    //--------------
    m->m[12] = 0.0;
    m->m[13] = 0.0;
    m->m[14] = 0.0;
    m->m[15] = 1.0;
    //--------------
    return;
}




void create4x4FRotationZXY(struct Matrix4x4OfFloats * m,float degreesX,float degreesY,float degreesZ)
{
    //---------------------------------------------------------
    float radiansX = (float) degreesX * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX = (float) cosf((float)radiansX);
    float sinX = (float) sinf((float)radiansX);
    //---------------------------------------------------------
    float radiansY = (float) degreesY * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosY = (float) cosf((float)radiansY);
    float sinY = (float) sinf((float)radiansY);
    //---------------------------------------------------------
    float radiansZ = (float) degreesZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radiansZ);
    float sinZ = (float) sinf((float)radiansZ);
    //---------------------------------------------------------
    //                       == RZ ==                                        == RX ==                                                  == RY ==
    //{ { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  } * { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} } * { { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }
    //https://www.wolframalpha.com/input?i=%7B+%7B+cosZ%2C+sinZ%2C+0+%7D+%2C+%7B+-sinZ%2C+cosZ+%2C+0%7D+%2C+%7B0%2C0%2C1%7D++%7D+*+%7B+%7B1%2C+0+%2C0%7D+%2C+%7B0%2C+cosX%2C+sinX%7D%2C+%7B0%2C+-sinX%2C+cosX%7D+%7D+*+%7B+%7B+cosY%2C+0%2C+-sinY+%7D%2C+%7B0%2C+1+%2C0+%7D%2C+%7B+sinY%2C+0+%2C+cosY+%7D+%7D
    //{{cosZ,sinZ,0,0},{-sinZ,cosZ,0,0},{0,0,1,0},{0,0,0,1}}*{{1,0,0,0},{0,cosX,sinX,0},{0,-sinX,cosX,0},{0,0,0,1}}*{{cosY,0,-sinY,0},{0,1,0,0},{sinY,0,cosY,0},{0,0,0,1}}
    //https://www.wolframalpha.com/input?key=&i=%7B%7BcosZ%2CsinZ%2C0%2C0%7D%2C%7B-sinZ%2CcosZ%2C0%2C0%7D%2C%7B0%2C0%2C1%2C0%7D%2C%7B0%2C0%2C0%2C1%7D%7D*%7B%7B1%2C0%2C0%2C0%7D%2C%7B0%2CcosX%2CsinX%2C0%7D%2C%7B0%2C-sinX%2CcosX%2C0%7D%2C%7B0%2C0%2C0%2C1%7D%7D*%7B%7BcosY%2C0%2C-sinY%2C0%7D%2C%7B0%2C1%2C0%2C0%7D%2C%7BsinY%2C0%2CcosY%2C0%7D%2C%7B0%2C0%2C0%2C1%7D%7D

    //1.50
    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //1.69 without memcpy
    //--------------
    //    Row 1
    //--------------
    m->m[0] = (sinX * sinY * sinZ) + (cosY * cosZ);
    m->m[1] = cosX * sinZ;
    m->m[2] = (sinX * cosY * sinZ) - (sinY * cosZ);
    //m->m[3] = 0.0;
    //--------------
    //    Row 2
    //--------------
    m->m[4] = (sinX * sinY * cosZ) - (cosY * sinZ);
    m->m[5] = cosX * cosZ;
    m->m[6] = (sinX * cosY * cosZ) + (sinY * sinZ);
    //m->m[7] = 0.0;
    //--------------
    //    Row 3
    //--------------
    m->m[8]  = cosX * sinY;
    m->m[9]  = -sinX;
    m->m[10] = cosX * cosY;
    //m->m[11] = 0.0;
    //--------------
    //    Row 4
    //--------------
    //m->m[12] = 0.0;
    //m->m[13] = 0.0;
    //m->m[14] = 0.0;
    //m->m[15] = 1.0;
    //--------------
    return;
}

void create4x4FRotationZYX(struct Matrix4x4OfFloats * m,float degreesX,float degreesY,float degreesZ)
{
    //---------------------------------------------------------
    float radiansX = (float) degreesX * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosX = (float) cosf((float)radiansX);
    float sinX = (float) sinf((float)radiansX);
    //---------------------------------------------------------
    float radiansY = (float) degreesY * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosY = (float) cosf((float)radiansY);
    float sinY = (float) sinf((float)radiansY);
    //---------------------------------------------------------
    float radiansZ = (float) degreesZ * ( (float) M_PI / 180.0 ); //degrees_to_radF(degrees);
    float cosZ = (float) cosf((float)radiansZ);
    float sinZ = (float) sinf((float)radiansZ);
    //---------------------------------------------------------
    //                       == RZ ==                                        == RY ==                                                  == RX ==
    //{ { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  } *  { { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }  * { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} }
    //https://www.wolframalpha.com/input?i=%7B+%7B+cosZ%2C+sinZ%2C+0+%7D+%2C+%7B+-sinZ%2C+cosZ+%2C+0%7D+%2C+%7B0%2C0%2C1%7D++%7D+*++%7B+%7B+cosY%2C+0%2C+-sinY+%7D%2C+%7B0%2C+1+%2C0+%7D%2C+%7B+sinY%2C+0+%2C+cosY+%7D+%7D++*+%7B+%7B1%2C+0+%2C0%7D+%2C+%7B0%2C+cosX%2C+sinX%7D%2C+%7B0%2C+-sinX%2C+cosX%7D+%7D+

    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //--------------
    //    Row 1
    //--------------
    m->m[0] = cosY * cosZ;
    m->m[1] = (sinX * sinY * cosZ) + (cosX * sinZ);
    m->m[2] = (sinX * sinZ) - (cosX * sinY * cosZ);
    //m->m[3] = 0.0;
    //--------------
    //    Row 2
    //--------------
    m->m[4] = -cosY * sinZ;
    m->m[5] = (cosX * cosZ) - (sinX * sinY * sinZ);
    m->m[6] = (cosX * sinY * sinZ) + (sinX * cosZ);
    //m->m[7] = 0.0;
    //--------------
    //    Row 3
    //--------------
    m->m[8]  = sinY;
    m->m[9]  = sinX * (-cosY);
    m->m[10] = cosX * cosY;
    //m->m[11] = 0.0;
    //--------------
    //    Row 4
    //--------------
    //m->m[12] = 0.0;
    //m->m[13] = 0.0;
    //m->m[14] = 0.0;
    //m->m[15] = 1.0;
    //--------------
    return;
}



void create4x4FMatrixFromEulerAnglesWithRotationOrder(struct Matrix4x4OfFloats * m,float degreesEulerX, float degreesEulerY, float degreesEulerZ,unsigned int rotationOrder)
{
  if (rotationOrder!=0)
  {
   char rXisIdentity = (degreesEulerX==0.0);
   char rYisIdentity = (degreesEulerY==0.0);
   char rZisIdentity = (degreesEulerZ==0.0);
   char numberOfIdentityMatrices = rXisIdentity+rYisIdentity+rZisIdentity;

   if (numberOfIdentityMatrices==3)
   {
     create4x4FIdentityMatrix(m);
   } else
   //if ( numberOfIdentityMatrices!=3 )
   {
    switch (rotationOrder)
    {
     case ROTATION_ORDER_XYZ :
       create4x4FRotationXYZ(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     case ROTATION_ORDER_XZY :
       create4x4FRotationXZY(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     case ROTATION_ORDER_YXZ :
       create4x4FRotationYXZ(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     case ROTATION_ORDER_YZX :
       create4x4FRotationYZX(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     case ROTATION_ORDER_ZXY :
       //This is the rotation order commonly used in all joints of the DAZ-Friendly CMU dataset ( https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release )
       //Speed this up and you get a speedup for MocapNET IK
       //{ {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} }
       //{ { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }
       //{ { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  } * { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} } * { { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }
       //4% speedup on IK by not using the multiplyThree4x4 Matrix call and using the precalculated version..!
       //fprintf(stderr,"create4x4FRotationZXY(%0.2f,%0.2f,%0.2f)\n",degreesEulerX,degreesEulerY,degreesEulerZ);
       create4x4FRotationZXY(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     case ROTATION_ORDER_ZYX :
       //This is the rotation order used in the LAFAN1 dataset ( https://github.com/ubisoft/ubisoft-laforge-animation-dataset )
       //And in the root hip rotation of the DAZ-Friendly CMU dataset ( https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release )
       //0.4% speedup on IK by not using the multiplyThree4x4 Matrix call and using the precalculated version..!
       create4x4FRotationZYX(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     case ROTATION_ORDER_RPY:
       doRPYTransformationF(
                            m,
                            degreesEulerZ,//roll,
                            degreesEulerY,//pitch
                            degreesEulerX //heading
                           );
     break;
     case ROTATION_ORDER_RODRIGUES :
       create4x4FMatrixFromRodriguez(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     default :
       fprintf(stderr,"create4x4MatrixFromEulerAnglesWithRotationOrderF: Error, Incorrect rotation type %u, returning Identity\n",rotationOrder);
       create4x4FIdentityMatrix(m);
     break;
    };
   } //else { create4x4FIdentityMatrix(m); }
  }  else { create4x4FIdentityMatrix(m); }

  return;
}


void create4x4FMatrixFromEulerAnglesWithRotationOrderOld(struct Matrix4x4OfFloats * m,float degreesEulerX, float degreesEulerY, float degreesEulerZ,unsigned int rotationOrder)
{
  if (rotationOrder!=0)
  {
   char rXisIdentity = (degreesEulerX==0.0);
   char rYisIdentity = (degreesEulerY==0.0);
   char rZisIdentity = (degreesEulerZ==0.0);

   if ( (!rXisIdentity) || (!rYisIdentity) || (!rZisIdentity) )
   {
    //Assuming the rotation axis are correct
    //rX,rY,rZ should hold our 4x4 rotation matrices
    struct Matrix4x4OfFloats rX;
    create4x4FRotationX(&rX,degreesEulerX);
    struct Matrix4x4OfFloats rY;
    create4x4FRotationY(&rY,degreesEulerY);
    struct Matrix4x4OfFloats rZ;
    create4x4FRotationZ(&rZ,degreesEulerZ);

   // ./BVHGUI2 --from dataset/MotionCapture/lafan1/dance2_subject2.bvh
   // ./BVHGUI2 --from dataset/headerWithHeadAndOneMotion.bvh

   switch (rotationOrder)
   {
     case ROTATION_ORDER_XYZ :
       //create4x4FRotationXYZ(m,degreesEulerX,degreesEulerY,degreesEulerZ);
       multiplyThree4x4FMatricesWithIdentityHints(m,&rX,rXisIdentity,&rY,rYisIdentity,&rZ,rZisIdentity);
     break;
     case ROTATION_ORDER_XZY :
       //create4x4FRotationXZY(m,degreesEulerX,degreesEulerY,degreesEulerZ);
       multiplyThree4x4FMatricesWithIdentityHints(m,&rX,rXisIdentity,&rZ,rZisIdentity,&rY,rYisIdentity);
     break;
     case ROTATION_ORDER_YXZ :
       //create4x4FRotationYXZ(m,degreesEulerX,degreesEulerY,degreesEulerZ);
       multiplyThree4x4FMatricesWithIdentityHints(m,&rY,rYisIdentity,&rX,rXisIdentity,&rZ,rZisIdentity);
     break;
     case ROTATION_ORDER_YZX :
       //create4x4FRotationYZX(m,degreesEulerX,degreesEulerY,degreesEulerZ);
       multiplyThree4x4FMatricesWithIdentityHints(m,&rY,rYisIdentity,&rZ,rZisIdentity,&rX,rXisIdentity);
     break;
     case ROTATION_ORDER_ZXY :
       //This is the rotation order commonly used in all joints of the DAZ-Friendly CMU dataset ( https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release )
       //Speed this up and you get a speedup for MocapNET IK
       //{ {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} }
       //{ { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }
       //{ { cosZ, sinZ, 0 } , { -sinZ, cosZ , 0} , {0,0,1}  } * { {1, 0 ,0} , {0, cosX, sinX}, {0, -sinX, cosX} } * { { cosY, 0, -sinY }, {0, 1 ,0 }, { sinY, 0 , cosY } }
       //4% speedup on IK by not using the multiplyThree4x4 Matrix call and using the precalculated version..!
       //create4x4FRotationZXY(m,degreesEulerX,degreesEulerY,degreesEulerZ);
       multiplyThree4x4FMatricesWithIdentityHints(m,&rZ,rZisIdentity,&rX,rXisIdentity,&rY,rYisIdentity);
     break;
     case ROTATION_ORDER_ZYX :
       //This is the rotation order used in the LAFAN1 dataset ( https://github.com/ubisoft/ubisoft-laforge-animation-dataset )
       //And in the root hip rotation of the DAZ-Friendly CMU dataset ( https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release )
       //0.4% speedup on IK by not using the multiplyThree4x4 Matrix call and using the precalculated version..!
       //create4x4FRotationZYX(m,degreesEulerX,degreesEulerY,degreesEulerZ);
       multiplyThree4x4FMatricesWithIdentityHints(m,&rZ,rZisIdentity,&rY,rYisIdentity,&rX,rXisIdentity);
     break;
     case ROTATION_ORDER_RPY:
       doRPYTransformationF(
                            m,
                            degreesEulerZ,//roll,
                            degreesEulerY,//pitch
                            degreesEulerX //heading
                           );
     break;
     case ROTATION_ORDER_RODRIGUES :
       create4x4FMatrixFromRodriguez(m,degreesEulerX,degreesEulerY,degreesEulerZ);
     break;
     default :
       fprintf(stderr,"create4x4MatrixFromEulerAnglesWithRotationOrderF: Error, Incorrect rotation type %u, returning Identity\n",rotationOrder);
       create4x4FIdentityMatrix(m);
     break;
    };


/*
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

*/
/*
    //Do check with new code, apparently i fixed it :)
    struct Matrix4x4OfFloats checkNew={0};
    create4x4FMatrixFromEulerAnglesWithRotationOrderNew(&checkNew,degreesEulerX,degreesEulerY,degreesEulerZ,rotationOrder);
    if (!are4x4FMatricesEqual(m->m,checkNew.m))
    {
      fprintf(stderr,"DISCREPANCY IN NEW 4x4 CODE for rotation order %u (%0.2f,%0.2f,%0.2f)!\n",rotationOrder,degreesEulerX,degreesEulerY,degreesEulerZ);
      print4x4FMatrix("New calculation",checkNew.m,1);
      print4x4FMatrix("Old calculation",m->m,1);
      exit(1);
    } else
    {
      fprintf(stderr,"New vs old 4x4 code matches..\n");
    }*/


    return;
   }
  }

  create4x4FIdentityMatrix(m);
  return;
}


void create4x4FRotationMatrix(struct Matrix4x4OfFloats * m , float angle, float x, float y, float z)
{
    float const DEG2RAD=(float) M_PI/180;
    float c = cosf(angle * DEG2RAD);
    float s = sinf(angle * DEG2RAD);
    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    float one_min_c = (1 - c);
    float x_mul_s = x * s;
    float y_mul_s = y * s;
    float z_mul_s = z * s;


    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //----------------------------------------------
    m->m[0] = xx * one_min_c + c;
    m->m[1] = xy * one_min_c - z_mul_s;
    m->m[2] = xz * one_min_c + y_mul_s;
    //m->m[3] = 0;
    m->m[4] = xy * one_min_c + z_mul_s;
    m->m[5] = yy * one_min_c + c;
    m->m[6] = yz * one_min_c - x_mul_s;
    //m->m[7] = 0;
    m->m[8] = xz * one_min_c - y_mul_s;
    m->m[9] = yz * one_min_c + x_mul_s;
    m->m[10]= zz * one_min_c + c;
    //m->m[11]= 0;
    //m->m[12]= 0;
    //m->m[13]= 0;
    //m->m[14]= 0;
    //m->m[15]= 1;
}

void create4x4FQuaternionMatrix(struct Matrix4x4OfFloats * m,float qX,float qY,float qZ,float qW)
{
    float yy2 = 2.0f * qY * qY;
    float xy2 = 2.0f * qX * qY;
    float xz2 = 2.0f * qX * qZ;
    float yz2 = 2.0f * qY * qZ;
    float zz2 = 2.0f * qZ * qZ;
    float wz2 = 2.0f * qW * qZ;
    float wy2 = 2.0f * qW * qY;
    float wx2 = 2.0f * qW * qX;
    float xx2 = 2.0f * qX * qX;

    memcpy(m->m,identityMatrix4x4,16*sizeof(float));
    //----------------------------------------------
    m->m[0]  = - yy2 - zz2 + 1.0f;
    m->m[1]  = xy2 + wz2;
    m->m[2]  = xz2 - wy2;
    //m->m[3]  = 0;
    m->m[4]  = xy2 - wz2;
    m->m[5]  = - xx2 - zz2 + 1.0f;
    m->m[6]  = yz2 + wx2;
    //m->m[7]  = 0;
    m->m[8]  = xz2 + wy2;
    m->m[9]  = yz2 - wx2;
    m->m[10] = - xx2 - yy2 + 1.0f;
    //m->m[11] = 0.0f;
    //m->m[12] = 0.0;
    //m->m[13] = 0.0;
    //m->m[14] = 0.0;
    //m->m[15] = 1.0f;
}


void create4x4FTranslationMatrix(struct Matrix4x4OfFloats * m , float x, float y, float z)
{
  //perf 0.95
  memcpy(m->m,identityMatrix4x4,16*sizeof(float));
  m->m[3]= x;
  m->m[7]= y;
  m->m[11]=z;

  /*
  #if OPTIMIZED
   //perf 1.35
   memset(m->m,0,16*sizeof(float));
   m->m[0] = 1.0;
   m->m[3] = x;
   m->m[5] = 1.0;
   m->m[7] = y;
   m->m[10] = 1.0;
   m->m[11] = z;
   m->m[15] = 1.0;
  #else
    create4x4FIdentityMatrix(m);
    // Translate slots.
    m->m[3] = x; m->m[7] = y; m->m[11] = z;
  #endif // OPTIMIZED
  */
}



void create4x4FScalingMatrix(struct Matrix4x4OfFloats * m, float scaleX, float scaleY, float scaleZ)
{
  memcpy(m->m,identityMatrix4x4,16*sizeof(float));
  m->m[0]  = scaleX;
  m->m[5]  = scaleY;
  m->m[10] = scaleZ;

  /*
   #if OPTIMIZED
   memset(m->m,0,16*sizeof(float));
   m->m[0]  = scaleX;
   m->m[5]  = scaleY;
   m->m[10] = scaleZ;
   m->m[15] = 1.0;
  #else
    if (m==0) { return ;}
    create4x4FIdentityMatrix(m);
    // Scale slots.
    m->m[0] = scaleX; m->m[5] = scaleY; m->m[10] = scaleZ;
  #endif // OPTIMIZED
  */
}

float det4x4FMatrix(const float * mat)
{
 const float * a = mat;

 float   detA  = a[I11] * a[I22] * a[I33]  * a[I44];
         detA += a[I11] * a[I23] * a[I34]  * a[I42];
         detA += a[I11] * a[I24] * a[I32]  * a[I43];

         detA += a[I12] * a[I21] * a[I34]  * a[I43];
         detA += a[I12] * a[I23] * a[I31]  * a[I44];
         detA += a[I12] * a[I24] * a[I33]  * a[I41];

         detA += a[I13] * a[I21] * a[I32]  * a[I44];
         detA += a[I13] * a[I22] * a[I34]  * a[I41];
         detA += a[I13] * a[I24] * a[I31]  * a[I42];

         detA += a[I14] * a[I21] * a[I33]  * a[I42];
         detA += a[I14] * a[I22] * a[I31]  * a[I43];
         detA += a[I14] * a[I23] * a[I32]  * a[I41];

  //FIRST PART DONE
         detA -= a[I11] * a[I22] * a[I34]  * a[I43];
         detA -= a[I11] * a[I23] * a[I32]  * a[I44];
         detA -= a[I11] * a[I24] * a[I33]  * a[I42];

         detA -= a[I12] * a[I21] * a[I33]  * a[I44];
         detA -= a[I12] * a[I23] * a[I34]  * a[I41];
         detA -= a[I12] * a[I24] * a[I31]  * a[I43];

         detA -= a[I13] * a[I21] * a[I34]  * a[I42];
         detA -= a[I13] * a[I22] * a[I31]  * a[I44];
         detA -= a[I13] * a[I24] * a[I32]  * a[I41];

         detA -= a[I14] * a[I21] * a[I32]  * a[I43];
         detA -= a[I14] * a[I22] * a[I33]  * a[I41];
         detA -= a[I14] * a[I23] * a[I31]  * a[I42];

 return detA;
}


int invert4x4FMatrix(struct Matrix4x4OfFloats * result,const struct Matrix4x4OfFloats * mat)
{
 const float * a = mat->m;
 float * b = result->m;
 float detA = det4x4FMatrix(mat->m);
 if (detA==0.0)
    {
      copy4x4FMatrix(result->m,mat->m);
      fprintf(stderr,"Matrix 4x4 cannot be inverted (det = 0)\n");
      return 0;
    }
 float one_div_detA = (float) 1 / detA;

 //FIRST LINE
 b[I11]  = a[I22] * a[I33] * a[I44] +  a[I23] * a[I34] * a[I42]  + a[I24] * a[I32] * a[I43];
 b[I11] -= a[I22] * a[I34] * a[I43] +  a[I23] * a[I32] * a[I44]  + a[I24] * a[I33] * a[I42];
 b[I11] *= one_div_detA;

 b[I12]  = a[I12] * a[I34] * a[I43] +  a[I13] * a[I32] * a[I44]  + a[I14] * a[I33] * a[I42];
 b[I12] -= a[I12] * a[I33] * a[I44] +  a[I13] * a[I34] * a[I42]  + a[I14] * a[I32] * a[I43];
 b[I12] *= one_div_detA;

 b[I13]  = a[I12] * a[I23] * a[I44] +  a[I13] * a[I24] * a[I42]  + a[I14] * a[I22] * a[I43];
 b[I13] -= a[I12] * a[I24] * a[I43] +  a[I13] * a[I22] * a[I44]  + a[I14] * a[I23] * a[I42];
 b[I13] *= one_div_detA;

 b[I14]  = a[I12] * a[I24] * a[I33] +  a[I13] * a[I22] * a[I34]  + a[I14] * a[I23] * a[I32];
 b[I14] -= a[I12] * a[I23] * a[I34] +  a[I13] * a[I24] * a[I32]  + a[I14] * a[I22] * a[I33];
 b[I14] *= one_div_detA;

 //SECOND LINE
 b[I21]  = a[I21] * a[I34] * a[I43] +  a[I23] * a[I31] * a[I44]  + a[I24] * a[I33] * a[I41];
 b[I21] -= a[I21] * a[I33] * a[I44] +  a[I23] * a[I34] * a[I41]  + a[I24] * a[I31] * a[I43];
 b[I21] *= one_div_detA;

 b[I22]  = a[I11] * a[I33] * a[I44] +  a[I13] * a[I34] * a[I41]  + a[I14] * a[I31] * a[I43];
 b[I22] -= a[I11] * a[I34] * a[I43] +  a[I13] * a[I31] * a[I44]  + a[I14] * a[I33] * a[I41];
 b[I22] *= one_div_detA;

 b[I23]  = a[I11] * a[I24] * a[I43] +  a[I13] * a[I21] * a[I44]  + a[I14] * a[I23] * a[I41];
 b[I23] -= a[I11] * a[I23] * a[I44] +  a[I13] * a[I24] * a[I41]  + a[I14] * a[I21] * a[I43];
 b[I23] *= one_div_detA;

 b[I24]  = a[I11] * a[I23] * a[I34] +  a[I13] * a[I24] * a[I31]  + a[I14] * a[I21] * a[I33];
 b[I24] -= a[I11] * a[I24] * a[I33] +  a[I13] * a[I21] * a[I34]  + a[I14] * a[I23] * a[I31];
 b[I24] *= one_div_detA;

 //THIRD LINE
 b[I31]  = a[I21] * a[I32] * a[I44] +  a[I22] * a[I34] * a[I41]  + a[I24] * a[I31] * a[I42];
 b[I31] -= a[I21] * a[I34] * a[I42] +  a[I22] * a[I31] * a[I44]  + a[I24] * a[I32] * a[I41];
 b[I31] *= one_div_detA;

 b[I32]  = a[I11] * a[I34] * a[I42] +  a[I12] * a[I31] * a[I44]  + a[I14] * a[I32] * a[I41];
 b[I32] -= a[I11] * a[I32] * a[I44] +  a[I12] * a[I34] * a[I41]  + a[I14] * a[I31] * a[I42];
 b[I32] *= one_div_detA;

 b[I33]  = a[I11] * a[I22] * a[I44] +  a[I12] * a[I24] * a[I41]  + a[I14] * a[I21] * a[I42];
 b[I33] -= a[I11] * a[I24] * a[I42] +  a[I12] * a[I21] * a[I44]  + a[I14] * a[I22] * a[I41];
 b[I33] *= one_div_detA;

 b[I34]  = a[I11] * a[I24] * a[I32] +  a[I12] * a[I21] * a[I34]  + a[I14] * a[I22] * a[I31];
 b[I34] -= a[I11] * a[I22] * a[I34] +  a[I12] * a[I24] * a[I31]  + a[I14] * a[I21] * a[I32];
 b[I34] *= one_div_detA;

 //FOURTH LINE
 b[I41]  = a[I21] * a[I33] * a[I42] +  a[I22] * a[I31] * a[I43]  + a[I23] * a[I32] * a[I41];
 b[I41] -= a[I21] * a[I32] * a[I43] +  a[I22] * a[I33] * a[I41]  + a[I23] * a[I31] * a[I42];
 b[I41] *= one_div_detA;

 b[I42]  = a[I11] * a[I32] * a[I43] +  a[I12] * a[I33] * a[I41]  + a[I13] * a[I31] * a[I42];
 b[I42] -= a[I11] * a[I33] * a[I42] +  a[I12] * a[I31] * a[I43]  + a[I13] * a[I32] * a[I41];
 b[I42] *= one_div_detA;

 b[I43]  = a[I11] * a[I23] * a[I42] +  a[I12] * a[I21] * a[I43]  + a[I13] * a[I22] * a[I41];
 b[I43] -= a[I11] * a[I22] * a[I43] +  a[I12] * a[I23] * a[I41]  + a[I13] * a[I21] * a[I42];
 b[I43] *= one_div_detA;

 b[I44]  = a[I11] * a[I22] * a[I33] +  a[I12] * a[I23] * a[I31]  + a[I13] * a[I21] * a[I32];
 b[I44] -= a[I11] * a[I23] * a[I32] +  a[I12] * a[I21] * a[I33]  + a[I13] * a[I22] * a[I31];
 b[I44] *= one_div_detA;


 #if PRINT_MATRIX_DEBUGGING
  print4x4FMatrix("Inverted Matrix From Source",a);
  print4x4FMatrix("Inverted Matrix To Target",b);
 #endif // PRINT_MATRIX_DEBUGGING

 return 1;
}


int transpose4x4FMatrix(float * mat)
{
  if (mat!=0)
  {
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */

  float tmp;
  tmp = mat[1]; mat[1]=mat[4];  mat[4]=tmp;
  tmp = mat[2]; mat[2]=mat[8];  mat[8]=tmp;
  tmp = mat[3]; mat[3]=mat[12]; mat[12]=tmp;


  tmp = mat[6]; mat[6]=mat[9]; mat[9]=tmp;
  tmp = mat[13]; mat[13]=mat[7]; mat[7]=tmp;
  tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;
  } else
  { //Believe it or not this is the fastest branch prediction :P
    return 0;
  }

 return 1;
}


int transpose4x4DMatrix(double * mat)
{
  if (mat!=0)
  {
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */

   double tmp;
   tmp = mat[1];  mat[1]=mat[4];   mat[4]=tmp;
   tmp = mat[2];  mat[2]=mat[8];   mat[8]=tmp;
   tmp = mat[3];  mat[3]=mat[12];  mat[12]=tmp;


   tmp = mat[6];  mat[6]=mat[9];   mat[9]=tmp;
   tmp = mat[13]; mat[13]=mat[7];  mat[7]=tmp;
   tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;
  } else
  { //Believe it or not this is the fastest branch prediction :P
    return 0;
  }

 return 1;
}



int multiplyTwo4x4DMatrices(double * result ,const double * matrixA ,const double * matrixB)
{
 if ( (matrixA!=0) && (matrixB!=0) && (result!=0) )
 {
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"Multiplying 4x4 A and B \n");
  print4x4DMatrix("A", matrixA);
  print4x4DMatrix("B", matrixB);
  #endif // PRINT_MATRIX_DEBUGGING

  //MULTIPLICATION_RESULT FIRST ROW
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[8]  + matrixA[3] * matrixB[12];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[9]  + matrixA[3] * matrixB[13];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[6]  + matrixA[2] * matrixB[10] + matrixA[3] * matrixB[14];
  result[3]=matrixA[0] * matrixB[3] + matrixA[1] * matrixB[7]  + matrixA[2] * matrixB[11] + matrixA[3] * matrixB[15];

  //MULTIPLICATION_RESULT SECOND ROW
  result[4]=matrixA[4] * matrixB[0] + matrixA[5] * matrixB[4]  + matrixA[6] * matrixB[8]  + matrixA[7] * matrixB[12];
  result[5]=matrixA[4] * matrixB[1] + matrixA[5] * matrixB[5]  + matrixA[6] * matrixB[9]  + matrixA[7] * matrixB[13];
  result[6]=matrixA[4] * matrixB[2] + matrixA[5] * matrixB[6]  + matrixA[6] * matrixB[10] + matrixA[7] * matrixB[14];
  result[7]=matrixA[4] * matrixB[3] + matrixA[5] * matrixB[7]  + matrixA[6] * matrixB[11] + matrixA[7] * matrixB[15];

  //MULTIPLICATION_RESULT THIRD ROW
  result[8] =matrixA[8] * matrixB[0] + matrixA[9] * matrixB[4]  + matrixA[10] * matrixB[8]   + matrixA[11] * matrixB[12];
  result[9] =matrixA[8] * matrixB[1] + matrixA[9] * matrixB[5]  + matrixA[10] * matrixB[9]   + matrixA[11] * matrixB[13];
  result[10]=matrixA[8] * matrixB[2] + matrixA[9] * matrixB[6]  + matrixA[10] * matrixB[10]  + matrixA[11] * matrixB[14];
  result[11]=matrixA[8] * matrixB[3] + matrixA[9] * matrixB[7]  + matrixA[10] * matrixB[11]  + matrixA[11] * matrixB[15];

  //MULTIPLICATION_RESULT FOURTH ROW
  result[12]=matrixA[12] * matrixB[0] + matrixA[13] * matrixB[4]  + matrixA[14] * matrixB[8]    + matrixA[15] * matrixB[12];
  result[13]=matrixA[12] * matrixB[1] + matrixA[13] * matrixB[5]  + matrixA[14] * matrixB[9]    + matrixA[15] * matrixB[13];
  result[14]=matrixA[12] * matrixB[2] + matrixA[13] * matrixB[6]  + matrixA[14] * matrixB[10]   + matrixA[15] * matrixB[14];
  result[15]=matrixA[12] * matrixB[3] + matrixA[13] * matrixB[7]  + matrixA[14] * matrixB[11]   + matrixA[15] * matrixB[15];

  #if PRINT_MATRIX_DEBUGGING
   print4x4DMatrix("AxB", result);
  #endif // PRINT_MATRIX_DEBUGGING

  } else
  { //Believe it or not this is the fastest branch prediction :P
    return 0;
  }

 return 1;
}


void multiplyThree4x4DMatrices(double * result ,const double * matrixA ,const double * matrixB ,const double * matrixC)
{
 if ( (matrixA!=0) && (matrixB!=0) && (matrixC!=0) && (result!=0) )
  {
   double tmp[16];
   multiplyTwo4x4DMatrices(tmp,matrixB,matrixC);
   multiplyTwo4x4DMatrices(result,matrixA,tmp);
  }
}



static inline void multiplyTwo4x4FMatrices_Naive(float * result,const float * matrixA,const float * matrixB)
{
  //if ( (matrixA!=0) && (matrixB!=0) && (result!=0) ) This gets called millions of times.. removing the check frees 0.43 clocks
  {
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"Multiplying 4x4 A and B \n");
  print4x4FMatrix("A", matrixA);
  print4x4FMatrix("B", matrixB);
  #endif // PRINT_MATRIX_DEBUGGING

  //MULTIPLICATION_RESULT FIRST ROW
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[8]  + matrixA[3] * matrixB[12];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[9]  + matrixA[3] * matrixB[13];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[6]  + matrixA[2] * matrixB[10] + matrixA[3] * matrixB[14];
  result[3]=matrixA[0] * matrixB[3] + matrixA[1] * matrixB[7]  + matrixA[2] * matrixB[11] + matrixA[3] * matrixB[15];

  //MULTIPLICATION_RESULT SECOND ROW
  result[4]=matrixA[4] * matrixB[0] + matrixA[5] * matrixB[4]  + matrixA[6] * matrixB[8]  + matrixA[7] * matrixB[12];
  result[5]=matrixA[4] * matrixB[1] + matrixA[5] * matrixB[5]  + matrixA[6] * matrixB[9]  + matrixA[7] * matrixB[13];
  result[6]=matrixA[4] * matrixB[2] + matrixA[5] * matrixB[6]  + matrixA[6] * matrixB[10] + matrixA[7] * matrixB[14];
  result[7]=matrixA[4] * matrixB[3] + matrixA[5] * matrixB[7]  + matrixA[6] * matrixB[11] + matrixA[7] * matrixB[15];

  //MULTIPLICATION_RESULT THIRD ROW
  result[8] =matrixA[8] * matrixB[0] + matrixA[9] * matrixB[4]  + matrixA[10] * matrixB[8]   + matrixA[11] * matrixB[12];
  result[9] =matrixA[8] * matrixB[1] + matrixA[9] * matrixB[5]  + matrixA[10] * matrixB[9]   + matrixA[11] * matrixB[13];
  result[10]=matrixA[8] * matrixB[2] + matrixA[9] * matrixB[6]  + matrixA[10] * matrixB[10]  + matrixA[11] * matrixB[14];
  result[11]=matrixA[8] * matrixB[3] + matrixA[9] * matrixB[7]  + matrixA[10] * matrixB[11]  + matrixA[11] * matrixB[15];

  //MULTIPLICATION_RESULT FOURTH ROW
  result[12]=matrixA[12] * matrixB[0] + matrixA[13] * matrixB[4]  + matrixA[14] * matrixB[8]    + matrixA[15] * matrixB[12];
  result[13]=matrixA[12] * matrixB[1] + matrixA[13] * matrixB[5]  + matrixA[14] * matrixB[9]    + matrixA[15] * matrixB[13];
  result[14]=matrixA[12] * matrixB[2] + matrixA[13] * matrixB[6]  + matrixA[14] * matrixB[10]   + matrixA[15] * matrixB[14];
  result[15]=matrixA[12] * matrixB[3] + matrixA[13] * matrixB[7]  + matrixA[14] * matrixB[11]   + matrixA[15] * matrixB[15];

  #if PRINT_MATRIX_DEBUGGING
   print4x4FMatrix("AxB", result);
  #endif // PRINT_MATRIX_DEBUGGING
  }

 return;
}



//https://software.intel.com/content/www/us/en/develop/articles/performance-of-classic-matrix-multiplication-algorithm-on-intel-xeon-phi-processor-system.html
static inline int multiplyTwo4x4FMatrices_CMMA(float * result ,const float * matrixA ,const float * matrixB)
{
  int res = 0;
  if ( (matrixA!=0) && (matrixB!=0) && (result!=0) )
  {
   for(int i = 0; i < 4; i += 1 )
	{
		for(int j = 0; j < 4; j += 1 )
		{
			float sum = 0.0;
			for(int k = 0; k < 4; k += 1 )
			{
				sum += (float)( matrixA[4*i+k] * matrixB[4*k+j] );
			}
			result[4*i+j] = sum;
		}
	}
	res=1;
  }

 return res;
}

/*
 * AVX2 + FMA 4x4 float matrix multiply.
 *
 * Strategy: compute two result rows per iteration using 256-bit registers.
 *
 *   YMM layout used here:
 *     bits   0-127 = result row  i
 *     bits 128-255 = result row i+1
 *
 *   For each B row k we duplicate it into both 128-bit halves of a YMM.
 *   We then broadcast A[i][k] into the low half and A[i+1][k] into the
 *   high half of a second YMM, then FMA the two together.
 *
 *   Total: 8 FMA256 instead of 16 MUL128 + 12 ADD128 (SSE2 path), giving
 *   ~2× throughput on Haswell and later.
 *
 * Alignment note: uses _mm256_loadu/storeu so no 32-byte alignment
 * requirement on the caller side (the structs are 16-byte aligned).
 * On Haswell+ there is no penalty when the data is cache-line aligned.
 *
 * Build flags required: -mavx2 -mfma
 */
static inline void multiplyTwo4x4FMatrices_AVX2(float * result,
                                                 const float * A,
                                                 const float * B)
{
#if INTEL_AVX2_OPTIMIZATIONS
    /* Load every row of B and duplicate it into both 128-bit lanes of a YMM.
     * _mm256_set_m128(hi, lo) stores  lo  in bits 0-127,  hi  in bits 128-255. */
    __m256 B0 = _mm256_set_m128(_mm_loadu_ps(&B[0]),  _mm_loadu_ps(&B[0]));
    __m256 B1 = _mm256_set_m128(_mm_loadu_ps(&B[4]),  _mm_loadu_ps(&B[4]));
    __m256 B2 = _mm256_set_m128(_mm_loadu_ps(&B[8]),  _mm_loadu_ps(&B[8]));
    __m256 B3 = _mm256_set_m128(_mm_loadu_ps(&B[12]), _mm_loadu_ps(&B[12]));

    /* --- Result rows 0 and 1 ---
     * lo lane: A_row0 x B,  hi lane: A_row1 x B
     * For step k: broadcast A[0][k] -> lo, A[1][k] -> hi */
    __m256 row01 =
        _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(A[4]), _mm_set1_ps(A[0])), B0,
        _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(A[5]), _mm_set1_ps(A[1])), B1,
        _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(A[6]), _mm_set1_ps(A[2])), B2,
        _mm256_mul_ps(  _mm256_set_m128(_mm_set1_ps(A[7]), _mm_set1_ps(A[3])), B3))));
    _mm256_storeu_ps(&result[0], row01);

    /* --- Result rows 2 and 3 ---
     * lo lane: A_row2 x B,  hi lane: A_row3 x B */
    __m256 row23 =
        _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(A[12]), _mm_set1_ps(A[8])),  B0,
        _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(A[13]), _mm_set1_ps(A[9])),  B1,
        _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(A[14]), _mm_set1_ps(A[10])), B2,
        _mm256_mul_ps(  _mm256_set_m128(_mm_set1_ps(A[15]), _mm_set1_ps(A[11])), B3))));
    _mm256_storeu_ps(&result[8], row23);
#else
    multiplyTwo4x4FMatrices_Naive(result, A, B);
#endif
}

//Interesting examples :
//https://github.com/microsoft/DirectXMath/blob/main/Inc/DirectXMathMatrix.inl
//https://github.com/romz-pl/matrix-matrix-multiply


#if INTEL_OPTIMIZATIONS
#define _MM_TRANSPOSE4_PS_HARDCODED(row0, row1, row2, row3) {       \
            __m128 tmp3, tmp2, tmp1, tmp0;                          \
                                                                    \
            tmp0   = _mm_shuffle_ps((row0), (row1), 0x44);          \
            tmp2   = _mm_shuffle_ps((row0), (row1), 0xEE);          \
            tmp1   = _mm_shuffle_ps((row2), (row3), 0x44);          \
            tmp3   = _mm_shuffle_ps((row2), (row3), 0xEE);          \
                                                                    \
            (row0) = _mm_shuffle_ps(tmp0, tmp1, 0x88);              \
            (row1) = _mm_shuffle_ps(tmp0, tmp1, 0xDD);              \
            (row2) = _mm_shuffle_ps(tmp2, tmp3, 0x88);              \
            (row3) = _mm_shuffle_ps(tmp2, tmp3, 0xDD);              \
        }
#endif


static inline void multiplyTwo4x4FMatrices_SSE3(float * result ,const float * matrixA ,const float * matrixB)
{
 //http://fhtr.blogspot.com/2010/02/4x4-float-matrix-multiplication-using.html
#if INTEL_OPTIMIZATIONS
 //Load Matrix A into registers
 __m128 matrixA_r0 = _mm_load_ps(&matrixA[0]);
 __m128 matrixA_r1 = _mm_load_ps(&matrixA[4]);
 __m128 matrixA_r2 = _mm_load_ps(&matrixA[8]);
 __m128 matrixA_r3 = _mm_load_ps(&matrixA[12]);

 /* //Instead of _MM_TRANSPOSE4_PS we can transpose the matrixB naively.. kills us @ 0.98% time..
 float __attribute__((aligned(16))) transposedMatrixB[16]={
      matrixB[0],matrixB[4],matrixB[8] ,matrixB[12],
      matrixB[1],matrixB[5],matrixB[9] ,matrixB[13],
      matrixB[2],matrixB[6],matrixB[10],matrixB[14],
      matrixB[3],matrixB[7],matrixB[11],matrixB[15]
      };*/

 //Load Matrix B into registers
 __m128 matrixB_r0 = _mm_load_ps(&matrixB[0]);
 __m128 matrixB_r1 = _mm_load_ps(&matrixB[4]);
 __m128 matrixB_r2 = _mm_load_ps(&matrixB[8]);
 __m128 matrixB_r3 = _mm_load_ps(&matrixB[12]);
 //Transpose matrixB in registers
 _MM_TRANSPOSE4_PS(matrixB_r0,matrixB_r1,matrixB_r2,matrixB_r3);

 //t0 - t4 are temporary registers

 //First Line!
 __m128 t0  = _mm_mul_ps(matrixA_r0,matrixB_r0);
 __m128 t1  = _mm_mul_ps(matrixA_r0,matrixB_r1);
 __m128 t2  = _mm_mul_ps(matrixA_r0,matrixB_r2);
 __m128 t3  = _mm_mul_ps(matrixA_r0,matrixB_r3);
        t0  = _mm_hadd_ps(t0,t1);
        t1  = _mm_hadd_ps(t2,t3);
 __m128 t4  = _mm_hadd_ps(t0,t1);
 //------------
 _mm_store_ps(&result[0], t4);

 //Second Line!
 t0  = _mm_mul_ps(matrixA_r1,matrixB_r0);
 t1  = _mm_mul_ps(matrixA_r1,matrixB_r1);
 t2  = _mm_mul_ps(matrixA_r1,matrixB_r2);
 t3  = _mm_mul_ps(matrixA_r1,matrixB_r3);
 t0  = _mm_hadd_ps(t0,t1);
 t1  = _mm_hadd_ps(t2,t3);
 t4  = _mm_hadd_ps(t0,t1);
 //------------
 _mm_store_ps(&result[4],t4);

 //Third Line!
 t0  = _mm_mul_ps(matrixA_r2,matrixB_r0);
 t1  = _mm_mul_ps(matrixA_r2,matrixB_r1);
 t2  = _mm_mul_ps(matrixA_r2,matrixB_r2);
 t3  = _mm_mul_ps(matrixA_r2,matrixB_r3);
 t0  = _mm_hadd_ps(t0,t1);
 t1  = _mm_hadd_ps(t2,t3);
 t4  = _mm_hadd_ps(t0,t1);
 //------------
 _mm_store_ps(&result[8],t4);

 //Forth Line!
 t0  = _mm_mul_ps(matrixA_r3,matrixB_r0);
 t1  = _mm_mul_ps(matrixA_r3,matrixB_r1);
 t2  = _mm_mul_ps(matrixA_r3,matrixB_r2);
 t3  = _mm_mul_ps(matrixA_r3,matrixB_r3);
 t0  = _mm_hadd_ps(t0,t1);
 t1  = _mm_hadd_ps(t2,t3);
 t4  = _mm_hadd_ps(t0,t1);
 //------------
 _mm_store_ps(&result[12],t4);
 #else
   multiplyTwo4x4FMatrices_Naive(result,matrixA,matrixB);
 #endif
}

//__attribute__((aligned(16)))
static inline void multiplyTwo4x4FMatrices_SSE2(float * result ,const float * matrixA ,const float * matrixB)
{
#if INTEL_OPTIMIZATIONS
    //https://software.intel.com/sites/landingpage/IntrinsicsGuide for more info

    //Load all rows to registers assuming that they are allocated using the __attribute__((aligned(16)))
    //We could load them using the _mm_loadu_ps however this performs 4x worse..!
    __m128 row1 = _mm_load_ps(&matrixB[0]);
    __m128 row2 = _mm_load_ps(&matrixB[4]);
    __m128 row3 = _mm_load_ps(&matrixB[8]);
    __m128 row4 = _mm_load_ps(&matrixB[12]);

    //First Column ------------------------
    //Broadcast the value in all elements
    __m128 brod1 = _mm_set1_ps(matrixA[4*0 + 0]);
    __m128 brod2 = _mm_set1_ps(matrixA[4*0 + 1]);
    __m128 brod3 = _mm_set1_ps(matrixA[4*0 + 2]);
    __m128 brod4 = _mm_set1_ps(matrixA[4*0 + 3]);
       //Add all required elements..
    __m128 row = _mm_add_ps(
                            _mm_add_ps(
                                       _mm_mul_ps(brod1, row1),
                                       _mm_mul_ps(brod2, row2)
                                      ),
                            _mm_add_ps(
                                       _mm_mul_ps(brod3, row3),
                                       _mm_mul_ps(brod4, row4)
                                      )
                           );
    _mm_store_ps(&result[4*0], row);

    //Second Column ------------------------
           brod1 = _mm_set1_ps(matrixA[4*1 + 0]);
           brod2 = _mm_set1_ps(matrixA[4*1 + 1]);
           brod3 = _mm_set1_ps(matrixA[4*1 + 2]);
           brod4 = _mm_set1_ps(matrixA[4*1 + 3]);
           row = _mm_add_ps(
                            _mm_add_ps(
                                       _mm_mul_ps(brod1, row1),
                                       _mm_mul_ps(brod2, row2)
                                      ),
                            _mm_add_ps(
                                       _mm_mul_ps(brod3, row3),
                                       _mm_mul_ps(brod4, row4)
                                      )
                           );
    _mm_store_ps(&result[4*1], row);

    //Third Column ------------------------
           brod1 = _mm_set1_ps(matrixA[4*2 + 0]);
           brod2 = _mm_set1_ps(matrixA[4*2 + 1]);
           brod3 = _mm_set1_ps(matrixA[4*2 + 2]);
           brod4 = _mm_set1_ps(matrixA[4*2 + 3]);
           row = _mm_add_ps(
                            _mm_add_ps(
                                       _mm_mul_ps(brod1, row1),
                                       _mm_mul_ps(brod2, row2)
                                      ),
                            _mm_add_ps(
                                       _mm_mul_ps(brod3, row3),
                                       _mm_mul_ps(brod4, row4)
                                      )
                           );
    _mm_store_ps(&result[4*2], row);

    //Fourth Column ------------------------
           brod1 = _mm_set1_ps(matrixA[4*3 + 0]);
           brod2 = _mm_set1_ps(matrixA[4*3 + 1]);
           brod3 = _mm_set1_ps(matrixA[4*3 + 2]);
           brod4 = _mm_set1_ps(matrixA[4*3 + 3]);
           row = _mm_add_ps(
                            _mm_add_ps(
                                       _mm_mul_ps(brod1, row1),
                                       _mm_mul_ps(brod2, row2)
                                      ),
                            _mm_add_ps(
                                       _mm_mul_ps(brod3, row3),
                                       _mm_mul_ps(brod4, row4)
                                      )
                           );
    _mm_store_ps(&result[4*3], row);
#else
   multiplyTwo4x4FMatrices_Naive(result ,matrixA,matrixB);
#endif
}



void multiplyTwo4x4FMatricesS(struct Matrix4x4OfFloats * result ,const struct Matrix4x4OfFloats * matrixA ,const struct Matrix4x4OfFloats * matrixB)
{
#if INTEL_AVX2_OPTIMIZATIONS
    multiplyTwo4x4FMatrices_AVX2(result->m,matrixA->m,matrixB->m);
#elif INTEL_OPTIMIZATIONS
    multiplyTwo4x4FMatrices_SSE2(result->m,matrixA->m,matrixB->m); //109.53 fps in the sven dataset
    ////multiplyTwo4x4FMatrices_SSE3(result->m,matrixA->m,matrixB->m); // 107.77 fps in the sven dataset
#else
   multiplyTwo4x4FMatrices_Naive(result->m,matrixA->m,matrixB->m);
#endif
}



void multiplyTwoRaw4x4FMatricesS(float * result ,const float * matrixA ,const float * matrixB)
{
#if INTEL_AVX2_OPTIMIZATIONS
    multiplyTwo4x4FMatrices_AVX2(result,matrixA,matrixB);
#elif INTEL_OPTIMIZATIONS
    multiplyTwo4x4FMatrices_SSE2(result,matrixA,matrixB); //109.53 fps in the sven dataset
    ////multiplyTwo4x4FMatrices_SSE3(result,matrixA,matrixB); // 107.77 fps in the sven dataset
#else
   multiplyTwo4x4FMatrices_Naive(result,matrixA,matrixB);
#endif
}


void multiplyTwo4x4FMatricesBuffered(struct Matrix4x4OfFloats * result ,const float * matrixA ,const float * matrixB)
{
  struct Matrix4x4OfFloats bufA;
   copy4x4FMatrix(bufA.m,matrixA);
  struct Matrix4x4OfFloats bufB;
   copy4x4FMatrix(bufB.m,matrixB);
   multiplyTwo4x4FMatricesS(result,&bufA,&bufB);
  return;
}


void multiplyThree4x4FMatrices_Naive(float * result ,const float * matrixA ,const float * matrixB ,const float * matrixC)
{
  if ( (matrixA!=0) && (matrixB!=0) && (matrixC!=0) && (result!=0) )
  {
   float tmp[16];
   multiplyTwo4x4FMatrices_Naive(tmp,matrixB,matrixC);
   multiplyTwo4x4FMatrices_Naive(result,matrixA, tmp);
  }
  return;
}

void multiplyThree4x4FMatrices(struct Matrix4x4OfFloats * result,const struct Matrix4x4OfFloats * matrixA,const struct Matrix4x4OfFloats * matrixB ,const struct Matrix4x4OfFloats * matrixC)
{
  if ( (matrixA!=0) && (matrixB!=0) && (matrixC!=0) && (result!=0) )
  {
   struct Matrix4x4OfFloats tmp;
   multiplyTwo4x4FMatricesS(&tmp,matrixB,matrixC);
   multiplyTwo4x4FMatricesS(result,matrixA,&tmp);
  }
  return;
}

enum THREE_4X4_MATRICES_CASES
{
    MAT_4x4_CASE_NO_ACTIVE_MATRICES=0,            // 0
    MAT_4x4_CASE_MATRIXA,                         // 1
    MAT_4x4_CASE_MATRIXB,                         // 2
    MAT_4x4_CASE_MATRIXA_AND_MATRIXB,             // 3
    MAT_4x4_CASE_MATRIXC,                         // 4
    MAT_4x4_CASE_MATRIXA_AND_MATRIXC,             // 5
    MAT_4x4_CASE_MATRIXB_AND_MATRIXC,             // 6
    MAT_4x4_CASE_MATRIXA_AND_MATRIXB_AND_MATRIXC  // 7
};

int multiplyThree4x4FMatricesWithIdentityHints(
                                                struct Matrix4x4OfFloats * result,
                                                const struct Matrix4x4OfFloats * matrixA,
                                                int matrixAIsIdentity,
                                                const struct Matrix4x4OfFloats * matrixB,
                                                int matrixBIsIdentity,
                                                const struct Matrix4x4OfFloats * matrixC,
                                                int matrixCIsIdentity
                                              )
{
  int res = 0;
  if ( (matrixA!=0) && (matrixB!=0) && (matrixC!=0) && (result!=0) )
  {
    unsigned char multiplicationCase  = MAT_4x4_CASE_NO_ACTIVE_MATRICES;
    //----------------------------------------------------
    multiplicationCase += (!matrixAIsIdentity) * MAT_4x4_CASE_MATRIXA;
    multiplicationCase += (!matrixBIsIdentity) * MAT_4x4_CASE_MATRIXB;
    multiplicationCase += (!matrixCIsIdentity) * MAT_4x4_CASE_MATRIXC;
    //fprintf(stderr,"Mult A:%u B:%u C:%u => %u\n",matrixAIsIdentity,matrixBIsIdentity,matrixCIsIdentity,multiplicationCase);
    //----------------------------------------------------
    switch (multiplicationCase)
    {
      //---------------------------------------------------------------------------------------------------------------------------
      case MAT_4x4_CASE_NO_ACTIVE_MATRICES              :  create4x4FIdentityMatrix(result);                          res = 1;  break;
      case MAT_4x4_CASE_MATRIXA                         :  copy4x4FMatrix(result->m,matrixA->m);                      res = 1;  break;
      case MAT_4x4_CASE_MATRIXB                         :  copy4x4FMatrix(result->m,matrixB->m);                      res = 1;  break;
      case MAT_4x4_CASE_MATRIXA_AND_MATRIXB             :  multiplyTwo4x4FMatricesS(result,matrixA,matrixB);          res = 1;  break;
      case MAT_4x4_CASE_MATRIXC                         :  copy4x4FMatrix(result->m,matrixC->m);                      res = 1;  break;
      case MAT_4x4_CASE_MATRIXA_AND_MATRIXC             :  multiplyTwo4x4FMatricesS(result,matrixA,matrixC);          res = 1;  break;
      case MAT_4x4_CASE_MATRIXB_AND_MATRIXC             :  multiplyTwo4x4FMatricesS(result,matrixB,matrixC);          res = 1;  break;
      case MAT_4x4_CASE_MATRIXA_AND_MATRIXB_AND_MATRIXC :  multiplyThree4x4FMatrices(result,matrixA,matrixB,matrixC); res = 1;  break;
      //---------------------------------------------------------------------------------------------------------------------------
      default:
        res = 0;
      //---------------------------------------------------------------------------------------------------------------------------
    };
  }
 return res;
}


void multiplyFour4x4FMatrices(
                              struct Matrix4x4OfFloats * result,
                              const struct Matrix4x4OfFloats * matrixA,
                              const struct Matrix4x4OfFloats * matrixB,
                              const struct Matrix4x4OfFloats * matrixC,
                              const struct Matrix4x4OfFloats * matrixD
                             )
{
  if ( (matrixA!=0) && (matrixB!=0) && (matrixC!=0) && (matrixD!=0) && (result!=0) )
  {
   struct Matrix4x4OfFloats tmpA;
   struct Matrix4x4OfFloats tmpB;
   multiplyTwo4x4FMatricesS(&tmpA,matrixC,matrixD);
   multiplyTwo4x4FMatricesS(&tmpB,matrixB,&tmpA);
   multiplyTwo4x4FMatricesS(result,matrixA,&tmpB);
  }
}


int transform3DNormalVectorUsing3x3FPartOf4x4FMatrix(float * resultPoint3D,struct Matrix4x4OfFloats * transformation4x4,float * point3D)
{
 if ( (resultPoint3D!=0) && (transformation4x4!=0) && (point3D!=0) )
 {
  if (point3D[3]!=0.0)
  {
    fprintf(stderr,"Error with W coordinate transform3DNormalVectorUsing3x3FPartOf4x4FMatrix , should be zero  \n");
    return 0;
  }

  float * m = transformation4x4->m;
  register float X=point3D[0],Y=point3D[1],W=point3D[2];
  /*
  What we want to do ( in mathematica )
   { {me0,me1,me2} , {me3,me4,me5} , {me6,me7,me8} } * { { X } , { Y } , { W } }

   This gives us

  {
    {me2 W + me0 X + me1 Y},
    {me5 W + me3 X + me4 Y},
    {me8 W + me6 X + me7 Y}
  }
*/

  float * me0=&m[e0] , * me1=&m[e1] , * me2=&m[e2]  ;  //m[e3]  ignored
  float * me3=&m[e4] , * me4=&m[e5] , * me5=&m[e6]  ;  //m[e7]  ignored
  float * me6=&m[e8] , * me7=&m[e9] , * me8=&m[e10] ;  //m[e11] ignored
  //       last line ignored since we only want 3x3


  resultPoint3D[0] =  (*me2) * W + (*me0) * X + (*me1) * Y;
  resultPoint3D[1] =  (*me5) * W + (*me3) * X + (*me4) * Y;
  resultPoint3D[2] =  (*me8) * W + (*me6) * X + (*me7) * Y;
  resultPoint3D[3] =  0;

 // Ok we have our results but now to normalize our vector
  if(resultPoint3D[2]!=0.0)
  {
   resultPoint3D[0]/=resultPoint3D[2];
   resultPoint3D[1]/=resultPoint3D[2];
   resultPoint3D[2]=1.0; //resultPoint3D[2]/=resultPoint3D[2];
  }

  return 1;
 }
 return 0;
}






/*
 * AVX2 + FMA 4x4 matrix * vec4 multiply.
 *
 * result = M * v  (row-major M, column vector v)
 *
 * Strategy: transpose M in registers so each 128-bit lane holds one
 * column of M, then FMA each column scaled by the matching element of v.
 * This avoids the horizontal-add (_mm_hadd_ps) used in the SSE3 path and
 * replaces it with FMA vertical operations which have better throughput.
 *
 * The W-divide (perspective division) is kept scalar since it rarely fires
 * in practice (W==1 for most BVH/pose usage) and mixing it with SIMD
 * would add a branch anyway.
 *
 * Build flags required: -mavx2 -mfma
 */
#if INTEL_AVX2_OPTIMIZATIONS
static inline void multiplyVectorWith4x4FMatrix_AVX2(float * result,
                                                      const float * M,
                                                      const float * v)
{
    /* Load the 4 rows of M */
    __m128 r0 = _mm_loadu_ps(&M[0]);
    __m128 r1 = _mm_loadu_ps(&M[4]);
    __m128 r2 = _mm_loadu_ps(&M[8]);
    __m128 r3 = _mm_loadu_ps(&M[12]);

    /* Transpose M in registers: after this r0..r3 are the 4 columns of M.
     * _MM_TRANSPOSE4_PS is defined in xmmintrin.h; immintrin.h also provides it. */
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

    /* result = col0*v[0] + col1*v[1] + col2*v[2] + col3*v[3]
     * Each _mm_set1_ps broadcast is a single VPBROADCASTSS on AVX2. */
    __m128 res = _mm_fmadd_ps(r0, _mm_set1_ps(v[0]),
                 _mm_fmadd_ps(r1, _mm_set1_ps(v[1]),
                 _mm_fmadd_ps(r2, _mm_set1_ps(v[2]),
                 _mm_mul_ps(  r3, _mm_set1_ps(v[3])))));

    _mm_storeu_ps(result, res);
}
#endif

#if INTEL_OPTIMIZATIONS
//__attribute__((aligned(16)))
void multiplyVectorWith4x4FMatrix_SSE(float * result ,const float * matrixA ,const float * point3D)
{
  //https://software.intel.com/sites/landingpage/IntrinsicsGuide for more info
 __m128 p  = _mm_load_ps(point3D);
 __m128 row1 = _mm_load_ps(&matrixA[0]);
 __m128 row2 = _mm_load_ps(&matrixA[4]);
 __m128 row3 = _mm_load_ps(&matrixA[8]);
 __m128 row4 = _mm_load_ps(&matrixA[12]);

 __m128 x = _mm_mul_ps(row1,p);
 __m128 y = _mm_mul_ps(row2,p);
 __m128 z = _mm_mul_ps(row3,p);
 __m128 w = _mm_mul_ps(row4,p);
 __m128 tmp1 = _mm_hadd_ps(x, y); // = [y2+y3, y0+y1, x2+x3, x0+x1]
 __m128 tmp2 = _mm_hadd_ps(z, w); // = [w2+w3, w0+w1, z2+z3, z0+z1]

 _mm_storeu_ps(result, _mm_hadd_ps(tmp1, tmp2)); // = [w0+w1+w2+w3, z0+z1+z2+z3, y0+y1+y2+y3, x0+x1+x2+x3]
}
#endif



int transform3DPointFVectorUsing4x4FMatrix_Naive(float * resultPoint3D,float * transformation4x4,float * point3D)
{
 int res = 0;
 if ( (resultPoint3D!=0) && (transformation4x4!=0) && (point3D!=0) )
 {
  // What we want to do ( in mathematica )
  // { {e0,e1,e2,e3} , {e4,e5,e6,e7} , {e8,e9,e10,e11} , {e12,e13,e14,e15} } * { { X } , { Y }  , { Z } , { W } }
  // https://www.wolframalpha.com/input?i=%7B+%7Ba%2Cb%2Cc%2Cd%7D+%2C+%7Be%2Cf%2Cg%2Ch%7D+%2C+%7Bi%2Cj%2Ck%2Cl%7D+%2C+%7Bm%2Cn%2Co%2Cp%7D+%7D+*+%7B+%7B+X+%7D+%2C+%7B+Y+%7D++%2C+%7B+Z+%7D+%2C+%7B+W+%7D+%7D

  // This gives us :
  //{
  //  {e3 W + e0 X + e1 Y + e2 Z},
  //  {e7 W + e4 X + e5 Y + e6 Z},
  //  {e11 W + e8 X + e9 Y + e10 Z},
  //  {e15 W + e12 X + e13 Y + e14 Z}
  //}
  float * m = transformation4x4;
  register float X=point3D[0],Y=point3D[1],Z=point3D[2],W=point3D[3];

  resultPoint3D[0] = m[e3]  * W + m[e0]  * X + m[e1]  * Y + m[e2]  * Z;
  resultPoint3D[1] = m[e7]  * W + m[e4]  * X + m[e5]  * Y + m[e6]  * Z;
  resultPoint3D[2] = m[e11] * W + m[e8]  * X + m[e9]  * Y + m[e10] * Z;
  resultPoint3D[3] = m[e15] * W + m[e12] * X + m[e13] * Y + m[e14] * Z;

  // Ok we have our results but now to normalize our vector
  if (resultPoint3D[3]!=0.0)
  {
     resultPoint3D[0]/=resultPoint3D[3];
     resultPoint3D[1]/=resultPoint3D[3];
     resultPoint3D[2]/=resultPoint3D[3];
     resultPoint3D[3]=1.0; // resultPoint3D[3]/=resultPoint3D[3];
     res = 1;
  } else
  {
     fprintf(stderr,"Error with W coordinate after multiplication of 3D Point with 4x4 Matrix\n");
     fprintf(stderr,"Input Point was %0.2f %0.2f %0.2f %0.2f \n",point3D[0],point3D[1],point3D[2],point3D[3]);
     fprintf(stderr,"Output Point was %0.2f %0.2f %0.2f %0.2f \n",resultPoint3D[0],resultPoint3D[1],resultPoint3D[2],resultPoint3D[3]);
     res = 0;
  }
 }
 return res;
}


//struct Vector4x1OfFloats
int transform3DPointFVectorUsing4x4FMatrix(struct Vector4x1OfFloats * resultPoint3D,struct Matrix4x4OfFloats * transformation4x4,struct Vector4x1OfFloats * point3D)
{
if ( (resultPoint3D!=0) && (transformation4x4!=0) && (point3D!=0) )
 {
#if INTEL_AVX2_OPTIMIZATIONS
  multiplyVectorWith4x4FMatrix_AVX2(resultPoint3D->m,transformation4x4->m,point3D->m);
  /* W-divide: fast path for the common W==1 case (all BVH/pose joints) */
  if (resultPoint3D->m[3] != 1.0f)
   { return normalize3DPointFVector(resultPoint3D->m); }
  return 1;
#elif INTEL_OPTIMIZATIONS
  multiplyVectorWith4x4FMatrix_SSE(resultPoint3D->m,transformation4x4->m,point3D->m);
#else
  //Use naive implementation
  return transform3DPointFVectorUsing4x4FMatrix_Naive(resultPoint3D->m,transformation4x4->m,point3D->m);
#endif

 }
return 0;
}

int normalize3DPointFVector(float * vec)
{
  if ( vec[3]==1.0 ) { return 1; } else
  if ( vec[3]!=0.0 )
  {
    vec[0]=vec[0]/vec[3];
    vec[1]=vec[1]/vec[3];
    vec[2]=vec[2]/vec[3];
    vec[3]=1.0; // vec[3]=vec[3]/vec[3];
    return 1;
  } else
  {
    fprintf(stderr,"normalize3DPointFVector cannot be normalized since element 3 is zero\n");
    return 0;
  }
}

int normalize3DPointDVector(double * vec)
{
  if ( vec[3]==1.0 ) { return 1; } else
  if ( vec[3]!=0.0 )
  {
    vec[0]=vec[0]/vec[3];
    vec[1]=vec[1]/vec[3];
    vec[2]=vec[2]/vec[3];
    vec[3]=1.0; // vec[3]=vec[3]/vec[3];
    return 1;
  } else
  {
    fprintf(stderr,"normalize3DPointDVector cannot be normalized since element 3 is zero\n");
    return 0;
  }
}


void doRPYTransformationF(
                         struct Matrix4x4OfFloats * m,
                         float  rollInDegrees,
                         float  pitchInDegrees,
                         float  yawInDegrees
                        )
{
    struct Matrix4x4OfFloats intermediateMatrixPitch;
    struct Matrix4x4OfFloats intermediateMatrixHeading;
    struct Matrix4x4OfFloats intermediateMatrixRoll;
    create4x4FRotationMatrix(&intermediateMatrixRoll   , rollInDegrees,      0.0,   0.0,   1.0);
    create4x4FRotationMatrix(&intermediateMatrixHeading, yawInDegrees,       0.0,   1.0,   0.0);
    create4x4FRotationMatrix(&intermediateMatrixPitch  , pitchInDegrees,     1.0,   0.0,   0.0);

    multiplyThree4x4FMatrices(
                              m ,
                              &intermediateMatrixRoll ,
                              &intermediateMatrixHeading ,
                              &intermediateMatrixPitch
                            );
}




void create4x4FModelTransformation(
                                   struct Matrix4x4OfFloats * m ,
                                   //Rotation Component
                                   float rotationX,//heading
                                   float rotationY,//pitch
                                   float rotationZ,//roll
                                   unsigned int rotationOrder,
                                   //Translation Component
                                   float x, float y, float z ,
                                   //Scaling Component
                                   float scaleX, float scaleY, float scaleZ
                                  )
{
    if (m==0) {return;}

    //fprintf(stderr,"Asked for a model transformation with RPY(%0.2f,%0.2f,%0.2f) ",rotationZ,rotationY,rotationX);
    //fprintf(stderr,"XYZ(%0.2f,%0.2f,%0.2f) ",x,y,z);
    //fprintf(stderr,"scaled(%0.2f,%0.2f,%0.2f)\n",scaleX,scaleY,scaleZ);

    //Translation matrix
    //----------------------------------------------------------
    int translationSpecified=0;
    struct Matrix4x4OfFloats intermediateMatrixTranslation;
    if ( (x!=0.0) || (y!=0.0) || (z!=0.0) )
    {
     create4x4FTranslationMatrix(
                                 &intermediateMatrixTranslation,
                                 x,
                                 y,
                                 z
                               );
      translationSpecified=1;
    } else
    {
      create4x4FIdentityMatrix(&intermediateMatrixTranslation); //This can be skipped to reduce cycles
    }
    //----------------------------------------------------------


    //Rotation matrix
    //----------------------------------------------------------
    int rotationSpecified=0;
    struct Matrix4x4OfFloats intermediateMatrixRotation;
    if ( (rotationX!=0.0) || (rotationY!=0.0) || (rotationZ!=0.0) )
    {
      if (rotationOrder>=ROTATION_ORDER_NUMBER_OF_NAMES)
        {
          fprintf(stderr,"create4x4FModelTransformation: wrong rotationOrder(%u)\n",rotationOrder);
          create4x4FIdentityMatrix(&intermediateMatrixRotation); //This can be skipped to reduce cycles
          rotationSpecified=0;
        } else
      if (rotationOrder==ROTATION_ORDER_RPY)
        {
          //This is the old way to do this rotation
          doRPYTransformationF(
                                &intermediateMatrixRotation,
                                rotationZ,//roll,
                                rotationY,//pitch
                                rotationX//heading
                              );
          rotationSpecified=1;
        } else
        {
          //fprintf(stderr,"Using new model transform code\n");
          create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                           &intermediateMatrixRotation ,
                                                           rotationX,
                                                           rotationY,
                                                           rotationZ,
                                                           rotationOrder
                                                          );
          rotationSpecified=1;
         }
    } else
    {
      create4x4FIdentityMatrix(&intermediateMatrixRotation); //This can be skipped to reduce cycles
    }
    //----------------------------------------------------------


    //Scale matrix
    //----------------------------------------------------------
    int scaleSpecified=0;
    struct Matrix4x4OfFloats intermediateScalingMatrix;
    if ( (scaleX!=1.0) || (scaleY!=1.0) || (scaleZ!=1.0) )
      {
        create4x4FScalingMatrix(&intermediateScalingMatrix,scaleX,scaleY,scaleZ);
        scaleSpecified=1;
      } else
      {
       create4x4FIdentityMatrix(&intermediateScalingMatrix); //This can be skipped to reduce cycles
      }

     //Re-use already optimized code-path
     multiplyThree4x4FMatricesWithIdentityHints(
                                                m,
                                                &intermediateMatrixTranslation,
                                                (translationSpecified==0),
                                                &intermediateMatrixRotation,
                                                (rotationSpecified==0),
                                                &intermediateScalingMatrix,
                                                (scaleSpecified==0)
                                               );
     return;
}





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
                                               )
{
    if (m==0) {return;}

    struct Matrix4x4OfFloats intermediateMatrixRX;
    struct Matrix4x4OfFloats intermediateMatrixRY;
    struct Matrix4x4OfFloats intermediateMatrixRZ;
    create4x4FRotationMatrix(&intermediateMatrixRX, rotationX_angleDegrees,  -1.0,   0.0,   0.0);
    create4x4FRotationMatrix(&intermediateMatrixRY, rotationY_angleDegrees,   0.0,  -1.0,   0.0);
    create4x4FRotationMatrix(&intermediateMatrixRZ, rotationZ_angleDegrees,   0.0,   0.0,  -1.0);

    struct Matrix4x4OfFloats intermediateMatrixRotation;
    multiplyThree4x4FMatrices(
                              &intermediateMatrixRotation ,
                              &intermediateMatrixRX ,
                              &intermediateMatrixRY ,
                              &intermediateMatrixRZ
                             );

    struct Matrix4x4OfFloats intermediateMatrixTranslation;
    create4x4FTranslationMatrix(
                               &intermediateMatrixTranslation,
                               -translationX_angleDegrees,
                               -translationY_angleDegrees,
                               -translationZ_angleDegrees
                              );

    multiplyTwo4x4FMatricesS(m,&intermediateMatrixRotation,&intermediateMatrixTranslation);
}
