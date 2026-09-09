/** @file quaternions.c
*   @brief  A Library that provides quaternion functionality
*   @author Ammar Qammaz (AmmarkoV)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quaternions.h"

//#define PI 3.141592653589793 precision anyone ? :P
#define PI 3.141592653589793238462643383279502884197
#define PI_DIV_180 0.01745329251
#define _180_DIV_PI 57.2957795131


#define USE_FAST_NORMALIZATION 0
#define USEATAN2 1
/* arctan and arcsin have a result between −π/2 and π/2. With three rotations between −π/2 and π/2 you can't have all possible orientations.
   We need to replace the arctan by atan2 to generate all the orientations. */


enum matrix4x4Enum
{
    m0_0=0,m0_1,m0_2,m0_3,
    m1_0,m1_1,m1_2,m1_3,
    m2_0,m2_1,m2_2,m2_3,
    m3_0,m3_1,m3_2,m3_3,
};

enum matrix3x3Enum
{
    m0=0,
    m1,m2,m3,m4,m5,m6,m7,m8
};


enum matrix3x3EnumTranspose
{
    mT0=0,mT3,mT6,
    mT1,mT4,mT7,
    mT2,mT5,mT8
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void handleQuaternionPackConvention(float qX,float qY,float qZ,float qW , float * packedQuaternionOutput,int quaternionConvention)
{
    switch (quaternionConvention)
    {
    case qWqXqYqZ  :
        packedQuaternionOutput[0]=qW;
        packedQuaternionOutput[1]=qX;
        packedQuaternionOutput[2]=qY;
        packedQuaternionOutput[3]=qZ;
        break;
    case qXqYqZqW :
        packedQuaternionOutput[0]=qX;
        packedQuaternionOutput[1]=qY;
        packedQuaternionOutput[2]=qZ;
        packedQuaternionOutput[3]=qW;
        break;

    default :
        fprintf(stderr,"Unhandled quaternion order given (%d) \n",quaternionConvention);
        break;
    }
}

void handleQuaternionUnpackConvention(float * packedQuaternionInput,float *qXOut,float *qYOut,float *qZOut,float *qWOut ,int quaternionConvention)
{
    switch (quaternionConvention)
    {
    case qWqXqYqZ  :
        *qWOut = packedQuaternionInput[0];
        *qXOut = packedQuaternionInput[1];
        *qYOut = packedQuaternionInput[2];
        *qZOut = packedQuaternionInput[3];
        break;
    case qXqYqZqW :
        *qXOut = packedQuaternionInput[0];
        *qYOut = packedQuaternionInput[1];
        *qZOut = packedQuaternionInput[2];
        *qWOut = packedQuaternionInput[3];
        break;

    default :
        fprintf(stderr,"Unhandled quaternion order given (%d) \n",quaternionConvention);
        break;
    }
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


int normalizeQuaternions(float *qX,float *qY,float *qZ,float *qW)
{
#if USE_FAST_NORMALIZATION
    // Works best when quat is already almost-normalized
    float f = (float) (3.0 - (((*qX) * (*qX)) + ( (*qY) * (*qY) ) + ( (*qZ) * (*qZ)) + ((*qW) * (*qW)))) / 2.0;
    *qX *= f;
    *qY *= f;
    *qZ *= f;
    *qW *= f;
#else
    float sqrtDown = (float) sqrt(((*qX) * (*qX)) + ( (*qY) * (*qY) ) + ( (*qZ) * (*qZ)) + ((*qW) * (*qW)));
    float f = (float) 1 / sqrtDown;
    *qX *= f;
    *qY *= f;
    *qZ *= f;
    *qW *= f;
#endif // USE_FAST_NORMALIZATION
    return 1;
}


float innerProductQuaternions(float qAX,float qAY,float qAZ,float qAW ,
                               float qBX,float qBY,float qBZ,float qBW)
{
    return (float) ((qAX * qBX) + (qAY * qBY)+ (qAZ * qBZ) + (qAW * qBW));
}


float anglesBetweenQuaternions(float qAX,float qAY,float qAZ,float qAW ,
                                float qBX,float qBY,float qBZ,float qBW)
{
    float rads= acos(innerProductQuaternions(qAX,qAY,qAZ,qAW,qBX,qBY,qBZ,qBW));

    return (float)  /*Why is the *2 needed ? */ 2* /*Why?*/ (rads * 180) / PI;
}


void multiplyQuaternions(float * qXOut,float * qYOut,float * qZOut,float * qWOut,
                         float qAX,float qAY,float qAZ,float qAW ,
                         float qBX,float qBY,float qBZ,float qBW)
{
    *qXOut = (qBX*qAW)+(qBW*qAX)+(qBZ*qAY)-(qBY*qAZ);
    *qYOut = (qBY*qAW)-(qBZ*qAX)+(qBW*qAY)+(qBX*qAZ);
    *qZOut = (qBZ*qAW)+(qBY*qAX)-(qBX*qAY)+(qBW*qAZ);
    *qWOut = (qBW*qAW)-(qBX*qAX)-(qBY*qAY)-(qBZ*qAZ);
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void euler2Quaternions(float * quaternions,float * euler,int quaternionConvention)
{
    //This conversion follows the rule euler X Y Z  to quaternions W X Y Z
    //Our input is degrees so we convert it to radians for the sin/cos functions
    float eX = (float) (euler[0] * PI) / 180;
    float eY = (float) (euler[1] * PI) / 180;
    float eZ = (float) (euler[2] * PI) / 180;

    //fprintf(stderr,"eX %f eY %f eZ %f\n",eX,eY,eZ);

    //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    //eX Roll  φ - rotation about the X-axis
    //eY Pitch θ - rotation about the Y-axis
    //eZ Yaw   ψ - rotation about the Z-axis

    float cosX2 = cos((float) eX/2); //cos(φ/2);
    float sinX2 = sin((float) eX/2); //sin(φ/2);
    float cosY2 = cos((float) eY/2); //cos(θ/2);
    float sinY2 = sin((float) eY/2); //sin(θ/2);
    float cosZ2 = cos((float) eZ/2); //cos(ψ/2);
    float sinZ2 = sin((float) eZ/2); //sin(ψ/2);

    float qX = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
    float qY = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
    float qZ = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
    float qW = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);

    handleQuaternionPackConvention(qX,qY,qZ,qW,quaternions,quaternionConvention);
}



void quaternions2Euler(float * euler,float * quaternions,int quaternionConvention)
{
    float qX,qY,qZ,qW;

    handleQuaternionUnpackConvention(quaternions,&qX,&qY,&qZ,&qW,quaternionConvention);

    //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    //e1 Roll  - rX: rotation about the X-axis
    //e2 Pitch - rY: rotation about the Y-axis
    //e3 Yaw   - rZ: rotation about the Z-axis

    //Shorthand to go according to http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles
    float q0=qW , q1 = qX , q2 = qY , q3 = qZ;
    float q0q1 = (float) q0*q1 , q2q3 = (float) q2*q3;
    float q0q2 = (float) q0*q2 , q3q1 = (float) q3*q1;
    float q0q3 = (float) q0*q3 , q1q2 = (float) q1*q2;


    float eXDenominator = ( 1.0 - 2.0 * (q1*q1 + q2*q2) );
    if (eXDenominator == 0.0 )
    {
        fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n");
        return;
    }
    float eYDenominator = ( 1.0 - 2.0 * ( q2*q2 + q3*q3) );
    if (eYDenominator == 0.0 )
    {
        fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n");
        return;
    }


#if USEATAN2
    /* arctan and arcsin have a result between −π/2 and π/2. With three rotations between −π/2 and π/2 you can't have all possible orientations.
       We need to replace the arctan by atan2 to generate all the orientations. */
    /*eX*/ euler[0] = atan2( (2.0 *  (q0q1 + q2q3)) , eXDenominator ) ;
    /*eY*/ euler[1] = asin( 2.0 * (q0q2 - q3q1));
    /*eZ*/ euler[2] = atan2( (2.0 * (q0q3 + q1q2)) ,  eYDenominator );
#else
#warning "Please note that the compiled output does not generate all possible orientations"
#warning "See : http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles"
#warning "You are strongly suggested to #define USEATAN2 1 in quaternions.c "
    /*eX*/ euler[0] = atan( (2.0 *  (q0q1 + q2q3)) / eXDenominator) ;
    /*eY*/ euler[1] = asin( 2.0 * (q0q2 - q3q1));
    /*eZ*/ euler[2] = atan( (2.0 * (q0q3 + q1q2)) /  eYDenominator );
#endif // USEATAN2


    //Our output is in radians so we convert it to degrees for the user

    //Go from radians back to degrees
    euler[0] = (euler[0] * 180) / PI;
    euler[1] = (euler[1] * 180) / PI;
    euler[2] = (euler[2] * 180) / PI;

}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void quaternionSlerp(float * qOut, float * q0,float * q1,float t)
{
    float product = (q0[0]*q1[0]) + (q0[1]*q1[1]) + (q0[2]*q1[2]) + (q0[3]*q1[3]); // -1,1)
    if (product<-1.0)
    {
        product=-1.0;
    }
    else if (product> 1.0)
    {
        product= 1.0;
    }

    float omega = acos(product);
    float absOmega = omega;
    if (absOmega<0.0)
    {
        absOmega=-1*absOmega;
    }
    if (absOmega < 1e-10)
    {
        if (omega<0.0)
        {
            omega = -1 * 1e-10;
        }
        else
        {
            omega = 1e-10;
        }
    }


    float som = sin(omega);
    float st0 = sin((1-t) * omega) / som;
    float st1 = sin(t * omega) / som;

    qOut[0] = q0[0]*st0 + q1[0]*st1;
    qOut[1] = q0[1]*st0 + q1[1]*st1;
    qOut[2] = q0[2]*st0 + q1[2]*st1;
    qOut[3] = q0[3]*st0 + q1[3]*st1;

    return;
}

void quaternion2Matrix3x3(float * matrix3x3,float * quaternions,int quaternionConvention)
{
    //http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    float qX,qY,qZ,qW;
    handleQuaternionUnpackConvention(quaternions,&qX,&qY,&qZ,&qW,quaternionConvention);

    float * m = matrix3x3;

    m[m0]=1 -(2*qY*qY) - (2*qZ*qZ); /*|*/  m[m1]=(2*qX*qY) - (2*qZ*qW);     /*|*/ m[m2]=(2*qX*qZ) + (2*qY*qW);
    m[m3]=(2*qX*qY) + (2*qZ*qW);    /*|*/  m[m4]=1 - (2*qX*qX) - (2*qZ*qZ); /*|*/ m[m5]=(2*qY*qZ) - (2*qX*qW);
    m[m6]=(2*qX*qZ) - (2*qY*qW);    /*|*/  m[m7]=(2*qY*qZ) + (2*qX*qW);     /*|*/ m[m8]=1 - (2*qX*qX) - (2*qY*qY);

    return ;
}



void quaternion2Matrix4x4(float * matrix4x4,float * quaternions,int quaternionConvention)
{
    //http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    float qX,qY,qZ,qW;
    handleQuaternionUnpackConvention(quaternions,&qX,&qY,&qZ,&qW,quaternionConvention);

    float * m = matrix4x4;

    float qYqY= qY*qY;
    float qZqZ= qZ*qZ;
    float qXqX= qX*qX;

    //-------------------------------------------------------------------------------------------------------------------------------------------
    m[0]=1.0 -(2.0*qYqY) - (2.0*qZqZ);   /*|*/  m[1]=(2.0*qX*qY) - (2.0*qW*qZ);      /*|*/ m[2]=(2.0*qX*qZ) + (2.0*qW*qY);      /*|*/ m[3]=0.0;
    m[4]=(2.0*qX*qY) + (2.0*qW*qZ);      /*|*/  m[5]=1.0 - (2.0*qXqX) - (2.0*qZqZ);  /*|*/ m[6]=(2.0*qY*qZ) - (2.0*qX*qW);      /*|*/ m[7]=0.0;
    m[8]=(2.0*qX*qZ) - (2.0*qW*qY);      /*|*/  m[9]=(2.0*qY*qZ) + (2.0*qW*qX);      /*|*/ m[10]=1.0 - (2.0*qXqX) - (2.0*qYqY); /*|*/ m[11]=0.0;
    m[12]=0.0;                           /*|*/  m[13]=0.0;                           /*|*/ m[14]=0.0;                           /*|*/ m[15]=1.0;
    //-------------------------------------------------------------------------------------------------------------------------------------------

    return ;
}


void matrix4x42Quaternion(float * quaternions,int quaternionConvention,float * matrix4x4)
{
//http://www.gamasutra.com/view/feature/131686/rotating_objects_using_quaternions.php
    float qX,qY,qZ,qW;
    float m[4][4];
    m[0][0] = matrix4x4[m0_0];    m[0][1] = matrix4x4[m0_1];    m[0][2] = matrix4x4[m0_2];    m[0][3] = matrix4x4[m0_3];
    m[1][0] = matrix4x4[m1_0];    m[1][1] = matrix4x4[m1_1];    m[1][2] = matrix4x4[m1_2];    m[1][3] = matrix4x4[m1_3];
    m[2][0] = matrix4x4[m2_0];    m[2][1] = matrix4x4[m2_1];    m[2][2] = matrix4x4[m2_2];    m[2][3] = matrix4x4[m2_3];
    m[3][0] = matrix4x4[m3_0];    m[3][1] = matrix4x4[m3_1];    m[3][2] = matrix4x4[m3_2];    m[3][3] = matrix4x4[m3_3];

    float  tr, s, q[4];
    int nxt[3] = {1, 2, 0};
    tr = m[0][0] + m[1][1] + m[2][2];
// check the diagonal
    if (tr > 0.0)
    {
        s = sqrt (tr + 1.0);
        qW = s / 2.0;
        s = 0.5 / s;
        qX = (m[1][2] - m[2][1]) * s;
        qY = (m[2][0] - m[0][2]) * s;
        qZ = (m[0][1] - m[1][0]) * s;
    }
    else
    {
        int i, j, k;
// diagonal is negative
        i = 0;
        if (m[1][1] > m[0][0]) i = 1;
        if (m[2][2] > m[i][i]) i = 2;
        j = nxt[i];
        k = nxt[j];
        s = sqrt ((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
        q[i] = s * 0.5;
        if (s != 0.0) s = 0.5 / s;
        q[3] = (m[j][k] - m[k][j]) * s;
        q[j] = (m[i][j] + m[j][i]) * s;
        q[k] = (m[i][k] + m[k][i]) * s;
        qX = q[0];
        qY = q[1];
        qZ = q[2];
        qW = q[3];
    }

 handleQuaternionPackConvention(qX,qY,qZ,qW ,quaternions,quaternionConvention);
}




void matrix3x32Quaternion(float * quaternions,int quaternionConvention,float * m3)
{
  float m4[16];

  m4[0]=m3[0];  m4[1]=m3[1];  m4[2]=m3[2];  m4[3]=0.0;
  m4[4]=m3[3];  m4[5]=m3[4];  m4[6]=m3[5];  m4[7]=0.0;
  m4[8]=m3[6];  m4[9]=m3[7];  m4[10]=m3[8]; m4[11]=0.0;
  m4[12]=0.0;   m4[13]=0.0;   m4[14]=0.0;   m4[15]=1.0;

 matrix4x42Quaternion(quaternions,quaternionConvention,m4);
}


void axisAngle2Quaternion(float * quaternionOutput,float xx,float yy,float zz,float a, int quaternionConvention)
{
    // Here we calculate the sin( theta / 2) once for optimization
    float aDeg= a*PI_DIV_180;
    float sin_aDegDiv2 = sin( aDeg / 2.0 );

    // Calculate the x, y and z of the quaternion
    float x = xx * sin_aDegDiv2;
    float y = yy * sin_aDegDiv2;
    float z = zz * sin_aDegDiv2;
    float w = cos( aDeg / 2.0 );

    // Calcualte the w value by cos( theta / 2 )
    normalizeQuaternions(&x,&y,&z,&w);

    handleQuaternionPackConvention(x,y,z,w,quaternionOutput,quaternionConvention);
}

/*
void fromAnyEulerRotationOrderToQuaternion()
 * {
 *
                   rX = tf2::Quaternion(tf2::Vector3(-1,0,0),degreesToRadians(-xRotation));
                   rY = tf2::Quaternion(tf2::Vector3(0,-1,0),degreesToRadians(-yRotation));
                   rZ = tf2::Quaternion(tf2::Vector3(0,0,-1),degreesToRadians(-zRotation));
                   qXYZW = rZ * rY * rX;
            or

                   rX = tf2::Quaternion(tf2::Vector3(-1,0,0),degreesToRadians(-xRotation));
                   rY = tf2::Quaternion(tf2::Vector3(0,-1,0),degreesToRadians(-yRotation));
                   rZ = tf2::Quaternion(tf2::Vector3(0,0,-1),degreesToRadians(-zRotation));
                   qXYZW = rZ * rX * rY;
 * }
*/

void quaternionRotate(float * quaternion , float rotX , float rotY, float rotZ , float angleDegrees , int quaternionConvention)
{
    float rotationQuaternion[4]={0};
    axisAngle2Quaternion(rotationQuaternion,rotX,rotY,rotZ,angleDegrees,quaternionConvention);


    normalizeQuaternions(&quaternion[pQX],&quaternion[pQY],&quaternion[pQZ],&quaternion[pQW]);
    float result[4]={0};
    multiplyQuaternions(&result[pQX],&result[pQY],&result[pQZ],&result[pQW],
                        rotationQuaternion[pQX],rotationQuaternion[pQY],rotationQuaternion[pQZ],rotationQuaternion[pQW],
                        quaternion[pQX],quaternion[pQY],quaternion[pQZ],quaternion[pQW] );


   quaternion[0]=result[0]; quaternion[1]=result[1]; quaternion[2]=result[2]; quaternion[3]=result[3];
}


//http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
void quaternionFromTwoVectors(float * quaternionOutput , float * vA , float * vB)
{
    float dotProductOfVectors = vA[0]*vB[0] + vA[1]*vB[1] + vA[2]*vB[2];
    float m = sqrt(2.f + 2.f * dotProductOfVectors);
    float wD = (1.f / m);

    //Doing CrossProduct
    quaternionOutput[0] = wD * ( vA[1]*vB[2] - vA[2]*vB[1] );
    quaternionOutput[1] = wD * ( vA[2]*vB[0] - vA[0]*vB[2] );
    quaternionOutput[2] = wD * ( vA[0]*vB[1] - vA[1]*vB[0] );
    quaternionOutput[3] = 0.5f * m; //qW
}


void generateRandomQuaternion(float * quaternionOutput)
{
 //Choose three points u, v, w ∈ [0,1] uniformly at random. A uniform, random quaternion is given by the simple expression:
 //h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))
 if (quaternionOutput!=0)
 {
     float u = ((float) rand() / (RAND_MAX));
     float v = ((float) rand() / (RAND_MAX));
     float w = ((float) rand() / (RAND_MAX));

     quaternionOutput[0] = sqrt(1-u) * sin(2 * PI * v );
     quaternionOutput[1] = sqrt(1-u) * cos(2 * PI * v );
     quaternionOutput[2] = sqrt(u)   * sin(2 * PI * w );
     quaternionOutput[3] = sqrt(u)   * sin(2 * PI * w );

     //Be 100% sure that the quaternion is normalized..
     normalizeQuaternions(&quaternionOutput[0],&quaternionOutput[1],&quaternionOutput[2],&quaternionOutput[3]);
 }
}

void stochasticRandomQuaternionWithLessThanAngleDistance(float * quaternionOutput,float * quaternionInput,int quaternionConvention,float angleDistance)
{
    if (angleDistance<=0.0)
    {
        fprintf(stderr,"stochasticRandomQuaternionWithLessThanAngleDistance is not possible with angle %0.2f\n",angleDistance);
        quaternionOutput[0]=quaternionInput[0];
        quaternionOutput[1]=quaternionInput[1];
        quaternionOutput[2]=quaternionInput[2];
        quaternionOutput[3]=quaternionInput[3];
        return;
    }

    //This is a stochastic call, if your angle distance is very small good luck..!
    float qIX=0.0,qIY=0.0,qIZ=0.0,qIW=1.0;
    handleQuaternionPackConvention(qIX,qIY,qIZ,qIW ,quaternionInput,quaternionConvention);

    float thisDistance;
    do
    {
     generateRandomQuaternion(quaternionOutput);
     float qOX=0.0,qOY=0.0,qOZ=0.0,qOW=1.0;
     handleQuaternionPackConvention(qOX,qOY,qOZ,qOW ,quaternionOutput,quaternionConvention);
     thisDistance = anglesBetweenQuaternions(
                                             qIX,qIY,qIZ,qIW,
                                             qOX,qOY,qOZ,qOW
                                            );
    }
    while (thisDistance>angleDistance);

}
