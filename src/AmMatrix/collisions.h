#ifndef COLLISIONS_H_INCLUDED
#define COLLISIONS_H_INCLUDED


/**
 * @brief  Definition for a plane Surface
 */
struct planeSurface
{
  float pos[3];
  float normal[3];
};

/**
 * @brief  Definition for a cube
 */
struct cubeVolume
{
  struct planeSurface planeX; float maxAbsoluteDistanceX;
  struct planeSurface planeY; float maxAbsoluteDistanceY;
  struct planeSurface planeZ; float maxAbsoluteDistanceZ;
};

struct approximateCubeList
{
  unsigned int MAXnumberOfCubes;
  unsigned int numberOfCubes;
  struct cubeVolume * cube;
};



int pointCollidesWithCubeList(float * point3D, struct approximateCubeList * cubeList);
int destroyCubeListSingle(struct approximateCubeList *cubeList);
int destroyCubeList(struct approximateCubeList ** cubeList);



#endif // COLLISIONS_H_INCLUDED
