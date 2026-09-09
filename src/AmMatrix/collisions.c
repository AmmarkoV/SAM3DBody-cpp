#include "collisions.h"
#include <stdio.h>
#include <stdlib.h>


float dotProduct(float p1[3] , float p2[3] )
{
    return p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2];
}


float  signedDistanceFromPlane(struct planeSurface * ps , float pN[3])
{
  float tempV[3];
  unsigned int i=0;
  for (i=0; i<3; i++)
  {
      tempV[i] = pN[i] - ps->pos[i];
  }

  return dotProduct(tempV, ps->normal);
}


int pointCollidesWithCubeList(float * point3D, struct approximateCubeList * cubeList)
{
  unsigned int cubeID=0;

  for (cubeID=0; cubeID<cubeList->numberOfCubes; cubeID++)
  {
    float dist =  signedDistanceFromPlane( &cubeList->cube[cubeID].planeX , point3D);
    if ( (dist>0.0) && (dist>cubeList->cube[cubeID].maxAbsoluteDistanceX) ) { return 0; } else
    if ( (dist<0.0) && (-1*dist>cubeList->cube[cubeID].maxAbsoluteDistanceX) ) { return 0; }

    dist =  signedDistanceFromPlane( &cubeList->cube[cubeID].planeY , point3D);
    if ( (dist>0.0) && (dist>cubeList->cube[cubeID].maxAbsoluteDistanceY) ) { return 0; } else
    if ( (dist<0.0) && (-1*dist>cubeList->cube[cubeID].maxAbsoluteDistanceY) ) { return 0; }

    dist =  signedDistanceFromPlane( &cubeList->cube[cubeID].planeZ , point3D);
    if ( (dist>0.0) && (dist>cubeList->cube[cubeID].maxAbsoluteDistanceZ) ) { return 0; } else
    if ( (dist<0.0) && (-1*dist>cubeList->cube[cubeID].maxAbsoluteDistanceZ) ) { return 0; }
  }

  return 1;
}



struct approximateCubeList * createCubeList(unsigned int maxCubes)
{
  struct approximateCubeList * newCubeList = (struct approximateCubeList *) malloc(sizeof(struct approximateCubeList));
  if (newCubeList!=0)
  {
     newCubeList->cube = (struct cubeVolume *) malloc(maxCubes * sizeof(struct cubeVolume) );
     if ( newCubeList->cube ==0 ) { free (newCubeList); return 0; }

     newCubeList->numberOfCubes=0;
     newCubeList->MAXnumberOfCubes=maxCubes;

     return newCubeList;
  }
  return 0;
}

//Hoa's box is 171 , 111 , 60
struct approximateCubeList * createSingleCubeObj(float sizeX,float sizeY,float sizeZ)
{
  struct approximateCubeList * retCubeList = createCubeList(1);

  retCubeList->cube[0].maxAbsoluteDistanceX = sizeX/2;
  retCubeList->cube[0].planeX.pos[0]=0.0;
  retCubeList->cube[0].planeX.pos[1]=0.0;
  retCubeList->cube[0].planeX.pos[2]=0.0;
  retCubeList->cube[0].planeX.normal[0]=1.0;
  retCubeList->cube[0].planeX.normal[1]=0.0;
  retCubeList->cube[0].planeX.normal[2]=0.0;

  retCubeList->cube[1].maxAbsoluteDistanceY = sizeY/2;
  retCubeList->cube[1].planeY.pos[0]=0.0;
  retCubeList->cube[1].planeY.pos[1]=0.0;
  retCubeList->cube[1].planeY.pos[2]=0.0;
  retCubeList->cube[1].planeY.normal[0]=0.0;
  retCubeList->cube[1].planeY.normal[1]=1.0;
  retCubeList->cube[1].planeY.normal[2]=0.0;

  retCubeList->cube[2].maxAbsoluteDistanceZ = sizeZ/2;
  retCubeList->cube[2].planeZ.pos[0]=0.0;
  retCubeList->cube[2].planeZ.pos[1]=0.0;
  retCubeList->cube[2].planeZ.pos[2]=0.0;
  retCubeList->cube[2].planeZ.normal[0]=0.0;
  retCubeList->cube[2].planeZ.normal[1]=0.0;
  retCubeList->cube[2].planeZ.normal[2]=1.0;

  return retCubeList;
}




int destroyCubeListSingle(struct approximateCubeList *cubeList)
{
  if (cubeList!=0)
  {
    if ( cubeList->cube !=0 ) { free(cubeList->cube); }
    free(cubeList);
    //cubeList=0;
  }
 return 1;
}


int destroyCubeList(struct approximateCubeList ** cubeList)
{
   destroyCubeListSingle(*cubeList);
   *cubeList=0;
   return 1;
}


