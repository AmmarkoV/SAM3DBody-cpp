/*
  THIS IS A STAGING FILE BASED ON OPENCV CODE  , TO BE REMOVED AND REPLACED WITH OWN WORK


M///////////////////////////////////////////////////////////////////////////////////////
00002 //
00003 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
00004 //
00005 //  By downloading, copying, installing or using the software you agree to this license.
00006 //  If you do not agree to this license, do not download, install,
00007 //  copy or use the software.
00008 //
00009 //
00010 //                        Intel License Agreement
00011 //                For Open Source Computer Vision Library
00012 //
00013 // Copyright (C) 2000, Intel Corporation, all rights reserved.
00014 // Third party copyrights are property of their respective owners.
00015 //
00016 // Redistribution and use in source and binary forms, with or without modification,
00017 // are permitted provided that the following conditions are met:
00018 //
00019 //   * Redistribution's of source code must retain the above copyright notice,
00020 //     this list of conditions and the following disclaimer.
00021 //
00022 //   * Redistribution's in binary form must reproduce the above copyright notice,
00023 //     this list of conditions and the following disclaimer in the documentation
00024 //     and/or other materials provided with the distribution.
00025 //
00026 //   * The name of Intel Corporation may not be used to endorse or promote products
00027 //     derived from this software without specific prior written permission.
00028 //
00029 // This software is provided by the copyright holders and contributors "as is" and
00030 // any express or implied warranties, including, but not limited to, the implied
00031 // warranties of merchantability and fitness for a particular purpose are disclaimed.
00032 // In no event shall the Intel Corporation or contributors be liable for any direct,
00033 // indirect, incidental, special, exemplary, or consequential damages
00034 // (including, but not limited to, procurement of substitute goods or services;
00035 // loss of use, data, or profits; or business interruption) however caused
00036 // and on any theory of liability, whether in contract, strict liability,
00037 // or tort (including negligence or otherwise) arising in any way out of
00038 // the use of this software, even if advised of the possibility of such damage.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ocvStaging.h"

#define Sgn(x)              ( (x)<0 ? -1:1 )    /* Sgn(0) = 1 ! */
#define REAL_ZERO(x)        ( (x) < 1e-8 && (x) > -1e-8)
#define MAX(a,b)            ( (a)<(b) ? b:a )



int icvSort( double *array, int length )
{
 int i, j, index;
 double swapd;

 if( !array || length < 1 ) return 0;

 for( i = 0; i < length - 1; i++ )
         {
           index = i;
           for( j = i + 1; j < length; j++ )
              {
               if( array[j] < array[index] )  index = j;
              }                       /* for */

            if( index - i )
             {
               swapd = array[i];
               array[i] = array[index];
               array[index] = swapd;
             }                       /* if */
          }                          /* for */

  return 1;
}


double
icvMedian( int *ml, int *mr, int num, double *F )
{
    double l1, l2, l3, d1, d2, value;
    double *deviation;
    int i, i3;

    if( !ml || !mr || !F )
        return -1;

    deviation = (double *) malloc( (num) * sizeof( double ));

    if( !deviation )
        return -1;

    for( i = 0, i3 = 0; i < num; i++, i3 += 3 )
    {

        l1 = F[0] * mr[i3] + F[1] * mr[i3 + 1] + F[2];
        l2 = F[3] * mr[i3] + F[4] * mr[i3 + 1] + F[5];
        l3 = F[6] * mr[i3] + F[7] * mr[i3 + 1] + F[8];

        d1 = (l1 * ml[i3] + l2 * ml[i3 + 1] + l3) / sqrt( l1 * l1 + l2 * l2 );

        l1 = F[0] * ml[i3] + F[3] * ml[i3 + 1] + F[6];
        l2 = F[1] * ml[i3] + F[4] * ml[i3 + 1] + F[7];
        l3 = F[2] * ml[i3] + F[5] * ml[i3 + 1] + F[8];

        d2 = (l1 * mr[i3] + l2 * mr[i3 + 1] + l3) / sqrt( l1 * l1 + l2 * l2 );

        deviation[i] = (double) (d1 * d1 + d2 * d2);
    }                           /* for */

    if( !icvSort( deviation, num ) )
    {

        free( deviation );
        return -1;
    }                           /* if */

    value = deviation[num / 2];
    free(deviation );
    return value;

}


int
icvSingularValueDecomposition( int M,
                               int N,
                               double *A,
                               double *W, int get_U, double *U, int get_V, double *V )
{
    int i = 0, j, k, l = 0, i1, k1, l1 = 0;
    int iterations, error = 0, jN, iN, kN, lN = 0;
    double *rv1;
    double c, f, g, h, s, x, y, z, scale, anorm;
    double af, ag, ah, t;
    int MN = M * N;
    int NN = N * N;

    /*  max_iterations - maximum number QR-iterations
       cc - reduces requirements to number stitch (cc>1)
     */

    int max_iterations = 100;
    double cc = 100;

    if( M < N )
        return N;

    rv1 = (double *) malloc( N * sizeof( double ));

    if( rv1 == 0 )
        return N;

    for( iN = 0; iN < MN; iN += N )
    {
        for( j = 0; j < N; j++ )
            U[iN + j] = A[iN + j];
    }                           /* for */

    /*  Adduction to bidiagonal type (transformations of reflection).
       Bidiagonal matrix is located in W (diagonal elements)
       and in rv1 (upperdiagonal elements)
     */

    g = 0;
    scale = 0;
    anorm = 0;

    for( i = 0, iN = 0; i < N; i++, iN += N )
    {

        l = i + 1;
        lN = iN + N;
        rv1[i] = scale * g;

        /*  Multiplyings on the left  */

        g = 0;
        s = 0;
        scale = 0;

        for( kN = iN; kN < MN; kN += N )
            scale += fabs( U[kN + i] );

        if( !REAL_ZERO( scale ))
        {

            for( kN = iN; kN < MN; kN += N )
            {

                U[kN + i] /= scale;
                s += U[kN + i] * U[kN + i];
            }                   /* for */

            f = U[iN + i];
            g = -sqrt( s ) * Sgn( f );
            h = f * g - s;
            U[iN + i] = f - g;

            for( j = l; j < N; j++ )
            {

                s = 0;

                for( kN = iN; kN < MN; kN += N )
                {

                    s += U[kN + i] * U[kN + j];
                }               /* for */

                f = s / h;

                for( kN = iN; kN < MN; kN += N )
                {

                    U[kN + j] += f * U[kN + i];
                }               /* for */
            }                   /* for */

            for( kN = iN; kN < MN; kN += N )
                U[kN + i] *= scale;
        }                       /* if */

        W[i] = scale * g;

        /*  Multiplyings on the right  */

        g = 0;
        s = 0;
        scale = 0;

        for( k = l; k < N; k++ )
            scale += fabs( U[iN + k] );

        if( !REAL_ZERO( scale ))
        {

            for( k = l; k < N; k++ )
            {

                U[iN + k] /= scale;
                s += (U[iN + k]) * (U[iN + k]);
            }                   /* for */

            f = U[iN + l];
            g = -sqrt( s ) * Sgn( f );
            h = f * g - s;
            U[i * N + l] = f - g;

            for( k = l; k < N; k++ )
                rv1[k] = U[iN + k] / h;

            for( jN = lN; jN < MN; jN += N )
            {
                s = 0;

                for( k = l; k < N; k++ )
                    s += U[jN + k] * U[iN + k];

                for( k = l; k < N; k++ )
                    U[jN + k] += s * rv1[k];

            }                   /* for */

            for( k = l; k < N; k++ )
                U[iN + k] *= scale;
        }                       /* if */

        t = fabs( W[i] );
        t += fabs( rv1[i] );
        anorm = MAX( anorm, t );
    }                           /* for */

    anorm *= cc;

    /*  accumulation of right transformations, if needed  */

    if( get_V )
    {

        for( i = N - 1, iN = NN - N; i >= 0; i--, iN -= N )
        {

            if( i < N - 1 )
            {

                /*  pass-by small g  */
                if( !REAL_ZERO( g ))
                {

                    for( j = l, jN = lN; j < N; j++, jN += N )
                        V[jN + i] = U[iN + j] / U[iN + l] / g;

                    for( j = l; j < N; j++ )
                    {

                        s = 0;

                        for( k = l, kN = lN; k < N; k++, kN += N )
                            s += U[iN + k] * V[kN + j];

                        for( kN = lN; kN < NN; kN += N )
                            V[kN + j] += s * V[kN + i];
                    }           /* for */
                }               /* if */

                for( j = l, jN = lN; j < N; j++, jN += N )
                {
                    V[iN + j] = 0;
                    V[jN + i] = 0;
                }               /* for */
            }                   /* if */

            V[iN + i] = 1;
            g = rv1[i];
            l = i;
            lN = iN;
        }                       /* for */
    }                           /* if */

    /*  accumulation of left transformations, if needed  */

    if( get_U )
    {

        for( i = N - 1, iN = NN - N; i >= 0; i--, iN -= N )
        {

            l = i + 1;
            lN = iN + N;
            g = W[i];

            for( j = l; j < N; j++ )
                U[iN + j] = 0;

            /*  pass-by small g  */
            if( !REAL_ZERO( g ))
            {

                for( j = l; j < N; j++ )
                {

                    s = 0;

                    for( kN = lN; kN < MN; kN += N )
                        s += U[kN + i] * U[kN + j];

                    f = s / U[iN + i] / g;

                    for( kN = iN; kN < MN; kN += N )
                        U[kN + j] += f * U[kN + i];
                }               /* for */

                for( jN = iN; jN < MN; jN += N )
                    U[jN + i] /= g;
            }
            else
            {

                for( jN = iN; jN < MN; jN += N )
                    U[jN + i] = 0;
            }                   /* if */

            U[iN + i] += 1;
        }                       /* for */
    }                           /* if */

    /*  Iterations QR-algorithm for bidiagonal matrixes
       W[i] - is the main diagonal
       rv1[i] - is the top diagonal, rv1[0]=0.
     */

    for( k = N - 1; k >= 0; k-- )
    {

        k1 = k - 1;
        iterations = 0;

        for( ;; )
        {

            /*  Cycle: checking a possibility of fission matrix  */
            for( l = k; l >= 0; l-- )
            {

                l1 = l - 1;

                if( REAL_ZERO( rv1[l] ) || REAL_ZERO( W[l1] ))
                    break;
            }                   /* for */

            if( !REAL_ZERO( rv1[l] ))
            {

                /*  W[l1] = 0,  matrix possible to fission
                   by clearing out rv1[l]  */

                c = 0;
                s = 1;

                for( i = l; i <= k; i++ )
                {

                    f = s * rv1[i];
                    rv1[i] = c * rv1[i];

                    /*  Rotations are done before the end of the block,
                       or when element in the line is finagle.
                     */

                    if( REAL_ZERO( f ))
                        break;

                    g = W[i];

                    /*  Scaling prevents finagling H ( F!=0!) */

                    af = fabs( f );
                    ag = fabs( g );

                    if( af < ag )
                        h = ag * sqrt( 1 + (f / g) * (f / g) );
                    else
                        h = af * sqrt( 1 + (f / g) * (f / g) );

                    W[i] = h;
                    c = g / h;
                    s = -f / h;

                    if( get_U )
                    {

                        for( jN = 0; jN < MN; jN += N )
                        {

                            y = U[jN + l1];
                            z = U[jN + i];
                            U[jN + l1] = y * c + z * s;
                            U[jN + i] = -y * s + z * c;
                        }       /* for */
                    }           /* if */
                }               /* for */
            }                   /* if */


            /*  Output in this place of program means,
               that rv1[L] = 0, matrix fissioned
               Iterations of the process of the persecution
               will be executed always for
               the bottom block ( from l before k ),
               with increase l possible.
             */

            z = W[k];

            if( l == k )
                break;

            /*  Completion iterations: lower block
               became trivial ( rv1[K]=0)  */

            if( iterations++ == max_iterations )
                return k;

            /*  Shift is computed on the lowest order 2 minor.  */

            x = W[l];
            y = W[k1];
            g = rv1[k1];
            h = rv1[k];

            /*  consequent fission prevents forming a machine zero  */
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h) / y;

            /*  prevented overflow  */
            if( fabs( f ) > 1 )
            {
                g = fabs( f );
                g *= sqrt( 1 + (1 / f) * (1 / f) );
            }
            else
                g = sqrt( f * f + 1 );

            f = ((x - z) * (x + z) + h * (y / (f + fabs( g ) * Sgn( f )) - h)) / x;
            c = 1;
            s = 1;

            for( i1 = l; i1 <= k1; i1++ )
            {

                i = i1 + 1;
                g = rv1[i];
                y = W[i];
                h = s * g;
                g *= c;

                /*  Scaling at calculation Z prevents its clearing,
                   however if F and H both are zero - pass-by of fission on Z.
                 */

                af = fabs( f );
                ah = fabs( h );

                if( af < ah )
                    z = ah * sqrt( 1 + (f / h) * (f / h) );

                else
                {

                    z = 0;
                    if( !REAL_ZERO( af ))
                        z = af * sqrt( 1 + (h / f) * (h / f) );
                }               /* if */

                rv1[i1] = z;

                /*  if Z=0, the rotation is free.  */
                if( !REAL_ZERO( z ))
                {

                    c = f / z;
                    s = h / z;
                }               /* if */

                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y *= c;

                if( get_V )
                {

                    for( jN = 0; jN < NN; jN += N )
                    {

                        x = V[jN + i1];
                        z = V[jN + i];
                        V[jN + i1] = x * c + z * s;
                        V[jN + i] = -x * s + z * c;
                    }           /* for */
                }               /* if */

                af = fabs( f );
                ah = fabs( h );

                if( af < ah )
                    z = ah * sqrt( 1 + (f / h) * (f / h) );
                else
                {

                    z = 0;
                    if( !REAL_ZERO( af ))
                        z = af * sqrt( 1 + (h / f) * (h / f) );
                }               /* if */

                W[i1] = z;

                if( !REAL_ZERO( z ))
                {

                    c = f / z;
                    s = h / z;
                }               /* if */

                f = c * g + s * y;
                x = -s * g + c * y;

                if( get_U )
                {

                    for( jN = 0; jN < MN; jN += N )
                    {

                        y = U[jN + i1];
                        z = U[jN + i];
                        U[jN + i1] = y * c + z * s;
                        U[jN + i] = -y * s + z * c;
                    }           /* for */
                }               /* if */
            }                   /* for */

            rv1[l] = 0;
            rv1[k] = f;
            W[k] = x;
        }                       /* for */

        if( z < 0 )
        {

            W[k] = -z;

            if( get_V )
            {

                for( jN = 0; jN < NN; jN += N )
                    V[jN + k] *= -1;
            }                   /* if */
        }                       /* if */
    }                           /* for */

    free( rv1 );

    return error;

}                               /* vm_SingularValueDecomposition */


/*
int icvLMedS 	( 	int *  	points1,
		            int *  	points2,
		            int  	numPoints,
		            double *  	fundamentalMatrix
	            )
{
    int sample, j, amount_samples, done;
    int amount_solutions;
    int ml7[21], mr7[21];

    double F_try[9 * 3];
    double F[9];
    double Mj, Mj_new;

    int i, num;

    int *ml;
    int *mr;
    int *new_ml;
    int *new_mr;
    int new_num;
    int error;

    error = 1;

    if( fundamentalMatrix == 0 )
        return 0;

    num = numPoints;

    if( num < 6 )
    {
        return 0;
    }                           // if

    ml = (int *) cvAlloc( sizeof( int ) * num * 3 );
    mr = (int *) cvAlloc( sizeof( int ) * num * 3 );

    for( i = 0; i < num; i++ )
    {

        ml[i * 3] = points1[i * 2];
        ml[i * 3 + 1] = points1[i * 2 + 1];

        ml[i * 3 + 2] = 1;

        mr[i * 3] = points2[i * 2];
        mr[i * 3 + 1] = points2[i * 2 + 1];

        mr[i * 3 + 2] = 1;
    }                           // for

    if( num > 7 )
    {

        Mj = -1;
        amount_samples = 1000;  //  -------  Must be changed !  ---------

        for( sample = 0; sample < amount_samples; sample++ )
        {

            icvChoose7( ml, mr, num, ml7, mr7 );
            icvPoint7( ml7, mr7, F_try, &amount_solutions );

            for( i = 0; i < amount_solutions / 9; i++ )
            {

                Mj_new = icvMedian( ml, mr, num, F_try + i * 9 );

                if( Mj_new >= 0 && (Mj == -1 || Mj_new < Mj) )
                {

                    for( j = 0; j < 9; j++ )
                    {

                        F[j] = F_try[i * 9 + j];
                    }           // for

                    Mj = Mj_new;
                }               // if
            }                   // for
        }                       // for

        if( Mj == -1 )
            return 0;

        done = icvBoltingPoints( ml, mr, num, F, Mj, &new_ml, &new_mr, &new_num );

        if( done == -1 )
        {

            cvFree( &mr );
            cvFree( &ml );
            return 0;
        }                       // if

        if( done > 7 )
            error = icvPoints8( new_ml, new_mr, new_num, F );

        cvFree( &new_mr );
        cvFree( &new_ml );

    }
    else
    {
        error = icvPoint7( ml, mr, F, &i );
    }                           // if

    if( error == 1 )
        error = icvRank2Constraint( F );

    for( i = 0; i < 3; i++ )
        for( j = 0; j < 3; j++ )
            fundamentalMatrix[i * 3 + j] = (float) F[i * 3 + j];

    return error;

}                               // icvLMedS
*/
