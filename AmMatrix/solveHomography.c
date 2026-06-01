#include "solveHomography.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrixTools.h"
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"


//http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
static inline float sqrt_fast_approximationAgain(const float x)
{
  union
  {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1<<29) + (u.i >> 1) - (1<<22);
  return u.x;
}

double distanceBetween2DPoints(double *x1,double *y1, double *x2, double *y2 )
{
    //sqrt_fast_approximation
  double dx,dy;

  if (*x1>=*x2) { dx=*x1-*x2; } else { dx=*x2-*x1; }
  if (*y1>=*y2) { dy=*y1-*y2; } else { dy=*y2-*y1; }

  double output = (double) sqrt_fast_approximationAgain( (dx * dx) + (dy * dy)  );

  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"Comparing Point ( %0.2f , %0.2f ) with ( %0.2f , %0.2f ) = error %0.2f \n",*x1,*y1,*x2,*y2,output);
  #endif // PRINT_MATRIX_DEBUGGING


  return output;
}

double rand_FloatRange(double a, double b)
{
return ((b-a)*((float)rand()/RAND_MAX))+a;
}


void randomize3x3HomographyMatrix(double * m)
{
  m[0]=rand_FloatRange(0.0,5.0);   m[1]=rand_FloatRange(-2.0,2.0);   m[2]=rand_FloatRange(700.0,2000.0);
  m[3]=rand_FloatRange(0.0,2.0);   m[4]=rand_FloatRange(0.0,22.0);   m[5]=rand_FloatRange(-2000.0,0.0);
  m[6]=rand_FloatRange(-2.0,2.0);   m[7]=rand_FloatRange(0.0,5.0);   m[8]=1.0;
}


double testHomographyError(double * homography , unsigned int pointsNum ,  double * pointsA,  double * pointsB)
{
  double currentError=0.0;
  double * pointsAPtr = pointsA;
  double * pointsBPtr = pointsB;

  double source[3]={0};
  double actualTarget[3]={0};
  double thisTarget[3]={0};

  unsigned int i=0;
  for (i=0; i<pointsNum; i++)
  {
    source[0] = *pointsAPtr; ++pointsAPtr;
    source[1] = *pointsAPtr; ++pointsAPtr;
    source[2] = 1.0;

    actualTarget[0] = *pointsBPtr; ++pointsBPtr;
    actualTarget[1] = *pointsBPtr; ++pointsBPtr;
    actualTarget[2] = 1.0;


    transform2DPointVectorUsing3x3Matrix(thisTarget,homography,source);

    currentError += distanceBetween2DPoints(&actualTarget[0],&actualTarget[1],&thisTarget[0],&thisTarget[1]);
  }
  return currentError;
}



double solvePNPHomography(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB)
{
   double * hypothesis = alloc3x3Matrix();    if (hypothesis ==0) { return 0 ; }
   unsigned int numberOfIterations=0;
   double bestError=10000 , currentError = 100;
   while ( ( bestError > 0.5 ) && ( numberOfIterations < 1000 ) )
   {
     //random3x3Matrix(hypothesis,0.0,200.0);
     randomize3x3HomographyMatrix(hypothesis);

     currentError =  testHomographyError( hypothesis , pointsNum , pointsA, pointsB);
     if (currentError<bestError)
     {
        bestError = currentError;

        copy3x3Matrix(result3x3Matrix,hypothesis);
     }

     ++numberOfIterations;
   }

  free3x3Matrix(&hypothesis);

  return bestError;
}





void testHomographySolver()
{
  double * F3x3 = alloc3x3Matrix();    if (F3x3 ==0) { return; }
  double * pointsA = (double *) malloc(sizeof(double) * 2 * 8);
  double * pointsB = (double *) malloc(sizeof(double) * 2 * 8);
  if ( (pointsA==0) || (pointsB==0) )
  {
    free(pointsA);   // free(NULL) is a no-op
    free(pointsB);
    free3x3Matrix(&F3x3);
    return;
  }
  memset(pointsA , 0 ,sizeof(double) * 2 * 8 );
  memset(pointsB , 0 ,sizeof(double) * 2 * 8 );


//          SOURCE FRAME POINTS                                 TARGET FRAME POINTS
//---------------------------------------------------------------------------------------------
  unsigned int i=0;
  pointsA[i*2+0]=34;    pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=33;    pointsB[i*2+1]=358; /* | | */ ++i;
  pointsA[i*2+0]=178;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=84;    pointsB[i*2+1]=374; /* | | */ ++i;
  pointsA[i*2+0]=320;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=139;   pointsB[i*2+1]=392; /* | | */ ++i;
  pointsA[i*2+0]=461;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=202;   pointsB[i*2+1]=410; /* | | */ ++i;
  pointsA[i*2+0]=605;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=271;   pointsB[i*2+1]=432; /* | | */ ++i;
  pointsA[i*2+0]=120;   pointsA[i*2+1]=312;  /* | | */  pointsB[i*2+0]=88;    pointsB[i*2+1]=318; /* | | */ ++i;
  pointsA[i*2+0]=219;   pointsA[i*2+1]=312;  /* | | */  pointsB[i*2+0]=134;   pointsB[i*2+1]=329; /* | | */ ++i;
  pointsA[i*2+0]=319;   pointsA[i*2+1]=312;  /* | | */  pointsB[i*2+0]=184;   pointsB[i*2+1]=342; /* | | */ ++i;
//---------------------------------------------------------------------------------------------



 double finalError = solvePNPHomography(F3x3 , i /*Number of points*/ ,  pointsA,  pointsB );

  print3x3DMatrix("Calculated matrix using 8 points", F3x3);

  print3x3DScilabMatrix("M",F3x3);

  printf("The error of this estimation was %0.2f\n",finalError);


  F3x3[0] = 3.369783338522472;     F3x3[1] = -1.637685275601417;   F3x3[2] = 851.0036476001653;
  F3x3[3] = -0.2783636300638685;   F3x3[4] =  15.54534472903452;   F3x3[5] = -2133.959529863233;
  F3x3[6] = -0.003793213664419078; F3x3[7] =  0.02530490689886264; F3x3[8] = 1;

  print3x3DMatrix("3x3 known good homography", F3x3);
  double err = testHomographyError(F3x3,i ,  pointsA,  pointsB);
  printf("known good homography brings error %0.2f \n",err);



  free3x3Matrix(&F3x3);
  free(pointsA);
  free(pointsB);

  return ;

}

