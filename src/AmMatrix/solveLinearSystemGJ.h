/** @file solveLinearSystemGJ.h
 *  @brief  how to calculate a fundamental Matrix by using a GaussJordan solver
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef SOLVELINEARSYSTEMGJ_H_INCLUDED
#define SOLVELINEARSYSTEMGJ_H_INCLUDED

int calculateFundamentalMatrix(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB );

void testGJSolver();


#endif // SOLVELINEARSYSTEMGJ_H_INCLUDED
