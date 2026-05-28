#include "solveLinearSystemGJ.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"
#include "solveHomography.h"


enum packedPointPrecalcs
{
  xBxA=0
 ,xByA
 ,xB
 ,yBxA
 ,yByA
 ,yB
 ,xA
 ,yA
 ,One
 ,Result
 ,ElementsNumber
};



enum packedPointPrecalcsFirstLine
{
  m_line1_minus_xA=0
 ,m_line1_minus_yA
 ,m_line1_minus_One_1
 ,m_line1_zero_1
 ,m_line1_zero_2
 ,m_line1_zero_3
 ,m_line1_xBxA
 ,m_line1_xByA
 ,m_line1_xB
 ,m_line1_Result
};

enum packedPointPrecalcsSecondLine
{
  m_line2_zero_1=0
 ,m_line2_zero_2
 ,m_line2_zero_3
 ,m_line2_minus_xA
 ,m_line2_minus_yA
 ,m_line2_minus_One_1
 ,m_line2_yBxA
 ,m_line2_yByA
 ,m_line2_yB
 ,m_line2_Result
};



void printSystemMathematicaJazz(double * mat,unsigned int totalLines)
{
  fprintf(stderr,"\n\n { ");
    unsigned int  i=0;
    for (i=0; i< totalLines; i++)
    {
     fprintf(stderr,"m = { %0.2f , ",mat[i*ElementsNumber + xBxA] );
     fprintf(stderr,"%0.2f , ",mat[i*ElementsNumber + xByA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + xB] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yBxA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yByA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yB] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + xA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + One] );
     fprintf(stderr," } " );
     if (i<totalLines-1) { fprintf(stderr," , "); }
    }
  fprintf(stderr,"} \n ");

  fprintf(stderr,"Equal @@ # & /@ Transpose[{m.{e11, e12, e13, e21, e22, e23, e31, e32, e33}, {0, 0, 0, 0, 0, 0, 0, 0}}]\n");
  fprintf(stderr,"Solve[%%]\n");

}



void printSystemMathematica(double * mat,unsigned int totalLines)
{
  fprintf(stderr,"\n\n m = { ");
    unsigned int  i=0;
    for (i=0; i< totalLines; i++)
    {
     fprintf(stderr," { %0.2f , ",mat[i*ElementsNumber + xBxA] );
     fprintf(stderr,"%0.2f , ",mat[i*ElementsNumber + xByA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + xB] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yBxA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yByA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yB] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + xA] );
     fprintf(stderr,"%0.2f ,",mat[i*ElementsNumber + yA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + One] );
     fprintf(stderr," } " );
     if (i<totalLines-1) { fprintf(stderr," , "); }
    }
  fprintf(stderr,"} \n ");


}

void printSystemScilab(double * mat,unsigned int totalLines)
{
//M = [8 1 6; 3 5 7; 4 9 2]
  fprintf(stderr,"M=[");
    unsigned int i=0;
    for (i=0; i< totalLines; i++)
    {
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xBxA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xByA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xB] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yBxA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yByA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yB] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yA] );
     fprintf(stderr,"%0.2f ;",mat[i*ElementsNumber + One] );
     //fprintf(stderr,"%f } , ",mat[i*ElementsNumber + Result] );
    }
  fprintf(stderr,"]\n");
}



void printSystemPlain(double * mat,char * label , unsigned int totalLines)
{
  fprintf(stderr,"System Printout of %s \n",label);
  fprintf(stderr,"---------------------------------------------\n");
    unsigned int i=0;
    for (i=0; i<totalLines; i++)
    {
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xBxA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xByA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xB] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yBxA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yByA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yB] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + xA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + yA] );
     fprintf(stderr,"%0.2f ",mat[i*ElementsNumber + One] );
     fprintf(stderr,"%f "   ,mat[i*ElementsNumber + Result] );
     fprintf(stderr,"\n");
    }
  fprintf(stderr,"---------------------------------------------\n");
}

int swapMatrixLines(double * mat , unsigned int line1, unsigned int line2)
{
  fprintf(stderr,"Swapping line %u with line %u \n",line1,line2);
  double tmp;
  unsigned int i=0;
  for (i=0; i<ElementsNumber; i++)
    {
      tmp = mat[line1*ElementsNumber + i];
      mat[line1*ElementsNumber + i] = mat[line2*ElementsNumber + i];
      mat[line2*ElementsNumber + i] = tmp;
    }
  return 1;
}

int makeSureNonZero(double * mat   , unsigned int activeLine , unsigned int totalLines )
{
  if ( ( mat[activeLine*ElementsNumber + activeLine] >=  0.0000001 ) ||
       ( mat[activeLine*ElementsNumber + activeLine] <=  -0.0000001 ) )   { return 1; }

  //If we are here it means our line has a zero leading element
  unsigned int line=0;
  fprintf(stderr,"makeSureNonZero : totalLines %u disregarded \n",totalLines);
  for (line=activeLine+1; line<ElementsNumber; line++)
    {
      if ( ( mat[line*ElementsNumber + activeLine] >=  0.0000001 )||
           ( mat[line*ElementsNumber + activeLine] <=  -0.0000001 ) )
              { return swapMatrixLines(mat,activeLine,line);  }
    }

  return 0;
}

int createBaseOne(double * mat   , unsigned int activeLine)
{ //Divide all elements of active line with the initial element to create a base zero
  double divisor = mat[activeLine*ElementsNumber + activeLine];
  if (divisor == 0.0) { return 0; }

  unsigned int i=0;
  for (i=0; i<ElementsNumber; i++)
    {
      mat[activeLine*ElementsNumber + i] /= divisor;
    }

  return 1;
}

int subtractBase(double * mat   , unsigned int activeLine , unsigned int totalLines )
{
  double newval =0.0;
  double multiplier = 0.0;
  unsigned int i=0;
  unsigned int line=0;


  for (line=0;/*activeLine;*/ line<totalLines; line++)
  {
   if (line!=activeLine)
   {
     multiplier = mat[line*ElementsNumber + activeLine];
     for (i=0; i<ElementsNumber; i++)
      {
        newval = (double) ( mat[activeLine*ElementsNumber + i] * multiplier ) ;
        mat[line*ElementsNumber + i] -= newval;
      }
   }
  }
  return 1;
}



/* #0  #1  #2  #3  #4  #5  #6   #7     #8
  ________________________________   ______
#0  1  xA  xB  xC  xD  xE  xF   xG   RESULT
#1      1  xH  xI  xJ  xK  xL   xM   RESULT
#2          1  xN  xO  xP  xQ   xR   RESULT
#3              1  xS  xT  xU   xV   RESULT
#4                  1  xX  xY   xZ   RESULT
#5                     1   x1   x2   RESULT = -x1 * 1 - x2 * ( -x3 * x4 )
#6                          1   x3   RESULT = -x3 * x4
#7                              x4   RESULT =  x4  */

int gatherResultOLD(double * result , double * mat  , unsigned int totalLines )
{
  double calculatedItem = 0.0;
  unsigned int line=0;
  unsigned int i=0;

  line=totalLines-1;
  while (line>0)
  {
    calculatedItem = 0.0;
    if (line<totalLines-1)
    {
       for (i=line+1; i<totalLines; i++)
       {
          calculatedItem -= mat[line*ElementsNumber + i];
       }
    } else
    {
     calculatedItem = mat[line*ElementsNumber + line];
    }

    //We have calculated the result for this raw so we save it
    mat[line*ElementsNumber + Result] = calculatedItem;
    //And we propagate it to previous lines
    for (i=0; i<totalLines; i++)
        {
         mat[i*ElementsNumber + line] *= calculatedItem;
        }

     --line;
  }

  //Store results in resulting matrix
  for (i=0; i<totalLines; i++)
        {
          result[i] = mat[i*ElementsNumber + Result];
        }
  return 1;
}

/* #0  #1  #2  #3  #4  #5  #6   #7     #8
  ________________________________   ______
#0  1   0   0   0   0   0   0    0   RESULT
#1      1   0   0   0   0   0    0   RESULT
#2          1   0   0   0   0    0   RESULT
#3              1   0   0   0    0   RESULT
#4                  1   0   0    0   RESULT
#5                     1    0    0   RESULT = -x1 * 1 - x2 * ( -x3 * x4 )
#6                          1    0   RESULT = -x3 * x4
#7                               1   RESULT =  x4  */

int gatherResult(double * result , double * mat  , unsigned int totalLines )
{
  signed int line=0,i=0,ok=0;

  line=totalLines-1;
  while (line>=0)
  {
    ok=0;
    for (i=0; i<ElementsNumber; i++)
        {
          if (i==line)
          {
            if ( mat[line*ElementsNumber+i]==1 ) { ++ok; }
          } else
          {
	        if ( (mat[line*ElementsNumber+i] > -0.00001) && (mat[line*ElementsNumber+i] < 0.00001) ) { ++ok; }
          }
        }


    if (ok>=ElementsNumber-1)
    {
        fprintf(stderr,"Line %u is ok , solution #%u is %0.2f\n",line,line,mat[line*ElementsNumber+Result-1]);
        result[line]=mat[line*ElementsNumber+Result-1];
    } else
    {
        fprintf(stderr,"Line %u is not ok ,  %u/%d oks \n",line,ok,ElementsNumber);
    }

    if (line==0) { break; } else
                 { --line; }
  }

  return 1;
}





int solveLinearSystemGJ(double * result , double * coefficients , unsigned int variables , unsigned int totalLines )
{
    fprintf(stderr,"solveLinearSystemGJ : Variables %u argument is not taken into account (?)",variables);

    char name[128]={0};
    //Make the system upper diagonal
    unsigned int i=0;
    for (i=0; i<totalLines; i++)
    {
      if (! makeSureNonZero(coefficients,i,totalLines) ) { fprintf(stderr,"Error making sure that we have a non zero element first ( %u ) \n", i ); break; }
      if (! createBaseOne(coefficients,i) )              { fprintf(stderr,"Error creating base one  ( %u ) \n", i ); break; }
      if (! subtractBase(coefficients,i,totalLines) )    { fprintf(stderr,"Error subtracting base  ( %u ) \n" , i ); break; }

      snprintf(name,sizeof(name),"Echeloned Step %u",i);
      printSystemPlain(coefficients,name,totalLines);
    }

    fprintf(stderr,"\n\n");
    printSystemPlain(coefficients,"Echeloned",totalLines);

    //Populate the results matrix
    return gatherResult(result,coefficients,totalLines);
}








int calculateFundamentalMatrix8PointMultipleView(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB )
{
    if (pointsNum<8) { fprintf(stderr,"calculateFundamentalMatrix8Point requires at least 8 points\n"); return 0; }

    //http://en.wikipedia.org/wiki/Eight-point_algorithm#Step_1:_Formulating_a_homogeneous_linear_equation
    //
    //           ( Xa )            ( Xb )                ( e11 e12 e13 )
    //  pointsA  ( Ya )    pointsB ( Yb )              E ( e21 e22 e23 )
    //           ( 1  )            ( 1  )                ( e31 e32 e33 )
    //
    //So what we do is convert this to a matrix for solving a linear system of 8 equations
    // Xb*Xa*e11   +   Xb*Ya*e12     +    Xb*e13    +    yB*xA*e21     +    yB*yA*e22    +    yB*e23    +    xA*e31    +    yA*e32    +      1*e33    =   0


    double * pxA , * pyA , * pxB , * pyB ;
    int elements=10;

    if (pointsNum > SIZE_MAX / ((size_t)elements * sizeof(double))) { return 0; }
    double * compiledPoints = (double *) calloc(pointsNum, (size_t)elements * sizeof(double));
    if (compiledPoints==0) { return 0; }

    unsigned int i=0;
    for (i=0; i< pointsNum; i++)
    {
      //Shortcut our vars
      pxA = &pointsA[i*2 + 0]; pyA = &pointsA[i*2 + 1];
      pxB = &pointsB[i*2 + 0]; pyB = &pointsB[i*2 + 1];
      fprintf(stderr,"Pair %u : Point A %f,%f Point B %f,%f \n",i,*pxA,*pyA,*pxB,*pyB);
      //Make the precalculations for each of the elements
      compiledPoints[i*elements + xBxA] = (double)  (*pxB) * (*pxA);
      compiledPoints[i*elements + xByA] = (double)  (*pxB) * (*pyA);
      compiledPoints[i*elements + xB]   = (double)  (*pxB) ;
      compiledPoints[i*elements + yBxA] = (double)  (*pyB) * (*pxA);
      compiledPoints[i*elements + yByA] = (double)  (*pyB) * (*pyA);
      compiledPoints[i*elements + yB]   = (double)  (*pyB);
      compiledPoints[i*elements + xA]   = (double)  (*pxA);
      compiledPoints[i*elements + yA]   = (double)  (*pyA);
      compiledPoints[i*elements + One]  = 1.0;
      compiledPoints[i*elements + Result] = 0.0;
    }

    fprintf(stderr,"\n\n");
    printSystemPlain(compiledPoints,"Original",pointsNum);

    //fprintf(stderr,"\n\n");
    printSystemMathematica(compiledPoints, pointsNum);
    //printSystemScilab(compiledPoints, pointsNum);

    solveLinearSystemGJ(result3x3Matrix,compiledPoints,elements,pointsNum);

    print3x3DMatrix("Result 3x3 Matrix",result3x3Matrix);

   free(compiledPoints);

   return 1;
}



int calculateFundamentalMatrix8Point(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB )
{
    if (pointsNum<8) { fprintf(stderr,"calculateFundamentalMatrix8Point requires at least 8 points\n"); return 0; }

    //http://en.wikipedia.org/wiki/Eight-point_algorithm#Step_1:_Formulating_a_homogeneous_linear_equation
    //
    //           ( Xa )            ( Xb )                ( e11 e12 e13 )
    //  pointsA  ( Ya )    pointsB ( Yb )              E ( e21 e22 e23 )
    //           ( 1  )            ( 1  )                ( e31 e32 e33 )
    //
    //So what we do is convert this to a matrix for solving a linear system of 8 equations
    // Xb*Xa*e11   +   Xb*Ya*e12     +    Xb*e13    +    yB*xA*e21     +    yB*yA*e22    +    yB*e23    +    xA*e31    +    yA*e32    +      1*e33    =   0


    double * pxA , * pyA , * pxB , * pyB ;
    int elements=10;

    if (pointsNum > SIZE_MAX / ((size_t)elements * sizeof(double))) { return 0; }
    double * compiledPoints = (double *) calloc(pointsNum, (size_t)elements * sizeof(double));
    if (compiledPoints==0) { return 0; }

    unsigned int i=0;
    for (i=0; i< pointsNum; i+=2)
    {
      //Shortcut our vars
      pxA = &pointsA[i*2 + 0]; pyA = &pointsA[i*2 + 1];
      pxB = &pointsB[i*2 + 0]; pyB = &pointsB[i*2 + 1];
      fprintf(stderr,"Pair %u : Point A %f,%f Point B %f,%f \n",i,*pxA,*pyA,*pxB,*pyB);


      //Make the precalculations for each of the elements
      compiledPoints[i*elements + m_line1_minus_xA] = (double)  (-1.0) * (*pxA);
      compiledPoints[i*elements + m_line1_minus_yA] = (double)  (-1.0) * (*pyA);
      compiledPoints[i*elements + m_line1_minus_One_1]   = (double)  (-1.0) ;
      compiledPoints[i*elements + m_line1_zero_1] = (double)  0.0;
      compiledPoints[i*elements + m_line1_zero_2] = (double)  0.0;
      compiledPoints[i*elements + m_line1_zero_3] = (double)  0.0;
      compiledPoints[i*elements + m_line1_xBxA]   = (double)  (*pxB)*(*pxA);
      compiledPoints[i*elements + m_line1_xByA]   = (double)  (*pxB)*(*pyA);
      compiledPoints[i*elements + m_line1_xB]  = (*pxB);
      compiledPoints[i*elements + m_line1_Result] = 0.0;

      //Make the precalculations for each of the elements
      compiledPoints[(i+1)*elements + m_line2_zero_1] = (double)  0.0;
      compiledPoints[(i+1)*elements + m_line2_zero_2] = (double)  0.0;
      compiledPoints[(i+1)*elements + m_line2_zero_3] = (double)  0.0;
      compiledPoints[(i+1)*elements + m_line2_minus_xA] = (double)  (-1.0) * (*pxA);
      compiledPoints[(i+1)*elements + m_line2_minus_yA] = (double)  (-1.0) * (*pyA);
      compiledPoints[(i+1)*elements + m_line2_minus_One_1] = (double)  (-1.0) ;
      compiledPoints[(i+1)*elements + m_line2_yBxA]   = (double)  (*pyB)*(*pxA);
      compiledPoints[(i+1)*elements + m_line2_yByA]   = (double)  (*pyB)*(*pyA);
      compiledPoints[(i+1)*elements + m_line2_yB]  = (*pyB);
      compiledPoints[(i+1)*elements + m_line2_Result] = 0.0;


    }

    fprintf(stderr,"\n\n");
    printSystemPlain(compiledPoints,"Original",pointsNum);

    //fprintf(stderr,"\n\n");
    printSystemMathematica(compiledPoints, pointsNum);
    //printSystemScilab(compiledPoints, pointsNum);

    solveLinearSystemGJ(result3x3Matrix,compiledPoints,elements,pointsNum);



   free(compiledPoints);

   return 1;
}





void testGJSolver()
{
  double * F3x3 = alloc3x3Matrix();    if (F3x3 ==0) { return; }
  double * pointsA = (double *) calloc(16, sizeof(double));
  double * pointsB = (double *) calloc(16, sizeof(double));


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


  calculateFundamentalMatrix8Point(F3x3 , i /*Number of points*/ ,  pointsA,  pointsB );

  print3x3DMatrix("Calculated matrix using 8 points", F3x3);

  print3x3DScilabMatrix("M",F3x3);


/*
  double err = testHomographyError(F3x3 , i ,  pointsA , pointsB);
  printf("result homography brings error %0.2f \n",err);




  F3x3[0] = 3.369783338522472;     F3x3[1] = -1.637685275601417;   F3x3[2] = 851.0036476001653;
  F3x3[3] = -0.2783636300638685;   F3x3[4] =  15.54534472903452;   F3x3[5] = -2133.959529863233;
  F3x3[6] = -0.003793213664419078; F3x3[7] =  0.02530490689886264; F3x3[8] = 1;

  print3x3DMatrix("3x3 known good homography", F3x3);
        err = testHomographyError(F3x3 , i , pointsA , pointsB);
  printf("known good homography brings error %0.2f \n",err);

*/

  free3x3Matrix(&F3x3);
  free(pointsA);
  free(pointsB);

  return ;

}





