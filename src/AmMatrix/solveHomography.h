#ifndef SOLVEPNPITERATIVE_H_INCLUDED
#define SOLVEPNPITERATIVE_H_INCLUDED


double solvePNPHomography(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB);

double testHomographyError(double * homography , unsigned int pointsNum ,  double * pointsA,  double * pointsB);

void testHomographySolver();

#endif // SOLVEPNPITERATIVE_H_INCLUDED
