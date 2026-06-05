/** @file model_loader_transform_joints.c
 *  @brief  Recursive node transformations for TRIModels part of
            https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
 *  @author Ammar Qammaz (AmmarkoV)
 */
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix4x4Tools.h"
#include "quaternions.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

static float _triTrans_degrees_to_rad(float degrees)
{
    return degrees * (M_PI /180.0 );
}

static void _triTrans_create4x4MatrixFromEulerAnglesZYX(float * m ,float eulX, float eulY, float eulZ)
{
    //roll = X , pitch = Y , yaw = Z
    float x = _triTrans_degrees_to_rad(eulX);
    float y = _triTrans_degrees_to_rad(eulY);
    float z = _triTrans_degrees_to_rad(eulZ);

	float cr = cos(z);
	float sr = sin(z);
	float cp = cos(y);
	float sp = sin(y);
	float cy = cos(x);
	float sy = sin(x);

	float srsp = sr*sp;
	float crsp = cr*sp;

	m[0] = (float) cr*cp;
	m[1] = (float) crsp*sy - sr*cy;
	m[2] = (float) crsp*cy + sr*sy;
	m[3] = 0.0;  // 4x4

	m[4] = (float) sr*cp;
	m[5] = (float) srsp*sy + cr*cy;
	m[6] = (float) srsp*cy - cr*sy;
	m[7] = 0.0;  // 4x4

	m[8] = (float) -sp;
	m[9] = (float) cp*sy;
	m[10]= (float) cp*cy;
	m[11]= 0.0;  // 4x4

     // 4x4 last row
	m[12]= 0.0;
	m[13]= 0.0;
	m[14]= 0.0;
	m[15]= 1.0;
}

/// Clamp a value to 0-255
int Clamp(int i)
{
  if (i < 0) return 0;
  if (i > 255) return 255;
  return i;
}

/// h is from 0-360
/// s,v values are 0-1
/// r,g,b values are 0-255
void HsvToRgb(float h,float S,float V, float * r, float * g, float * b)
{
  // ######################################################################
  // T. Nathan Mundhenk
  // mundhenk@usc.edu
  // C/C++ Macro HSV to RGB

  float H = h;
  while (H < 0) { H += 360; };
  while (H >= 360) { H -= 360; };
  float R, G, B;
  if (V <= 0)
    { R = G = B = 0; }
  else if (S <= 0)
  {
    R = G = B = V;
  }
  else
  {
    float hf = H / 60.0;
    int i = (int) hf;
    float f = hf - i;
    float pv = V * (1 - S);
    float qv = V * (1 - S * f);
    float tv = V * (1 - S * (1 - f));
    switch (i)
    {
      // Red is the dominant color
      case 0:  R = V;  G = tv; B = pv; break;

      // Green is the dominant color
      case 1:  R = qv; G = V;  B = pv; break;
      case 2:  R = pv; G = V;  B = tv; break;

      // Blue is the dominant color
      case 3:  R = pv; G = qv; B = V; break;
      case 4:  R = tv; G = pv; B = V; break;

      // Red is the dominant color
      case 5:  R = V; G = pv; B = qv; break;

      // Just in case we overshoot on our math by a little, we put these here. Since its a switch it won't slow us down at all to put these here.
      case 6:  R = V; G = tv; B = pv; break;
      case -1: R = V; G = pv; B = qv; break;

      // The color is not defined, we should throw an error.
      default:
        //LFATAL("i Value error in Pixel conversion, Value is %d", i);
        R = G = B = V; // Just pretend its black/white
        break;
    }
  }
  *r = (float) Clamp((int)(R * 255.0));
  *g = (float) Clamp((int)(G * 255.0));
  *b = (float) Clamp((int)(B * 255.0));
}

void getDistinctColor3F_ForID(unsigned int id,unsigned maxID , float *oR,float *oG,float *oB)
{
  unsigned int sCoef=10;
  unsigned int vCoef=40;

  unsigned int hStep = (unsigned int) 360/maxID;
  unsigned int sStep = (unsigned int) sCoef/maxID;
  unsigned int vStep = (unsigned int) vCoef/maxID;

  // assumes hue [0, 360), saturation [0, 100), lightness [0, 100)
  float h = id * hStep ;
  float s = (100-sCoef) + ( (sStep-sCoef) * sStep );
  float v = (100-vCoef) + ( (vStep-vCoef) * vStep );

  HsvToRgb(h, (float) s/100, (float) v/100 , oR , oG , oB);

  *oR = (float) *oR / 255;
  *oG = (float) *oG / 255;
  *oB = (float) *oB / 255;
}

float * generatePalette(struct TRI_Model * in)
{
  unsigned int maxNumBones = in->header.numberOfBones;
  fprintf(stderr,"generating palette for model with %u bones \n",maxNumBones);
  unsigned int i=0 ;
  float * gp = (float*) malloc(sizeof(float)*maxNumBones * 3);

  if (gp!=0)
  {
   for (i=0; i<maxNumBones; i++)
   {
     getDistinctColor3F_ForID(
                              i,
                              maxNumBones,
                              &gp[3*i+0],
                              &gp[3*i+1],
                              &gp[3*i+2]
                             );
   }
  }

 return gp;
}
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
struct TRI_Bones_Per_Vertex * allocTransformTRIBonesToVertexBoneFormat(struct TRI_Model * in)
{
  struct TRI_Bones_Per_Vertex * out = (struct TRI_Bones_Per_Vertex *) malloc(sizeof(struct TRI_Bones_Per_Vertex));
  if (out==0) { return 0; }

  out->numberOfBones = in->header.numberOfBones;
  out->numberOfVertices = in->header.numberOfVertices;
  out->bonesPerVertex = (struct TRI_Bones_Per_Vertex_Vertice_Item *)   malloc(sizeof(struct TRI_Bones_Per_Vertex_Vertice_Item) *  in->header.numberOfVertices);
  if (out->bonesPerVertex==0) { free(out); return 0; }

  memset(out->bonesPerVertex,0,sizeof(struct TRI_Bones_Per_Vertex_Vertice_Item) *  in->header.numberOfVertices);

  unsigned int b=0 , w=0 , outOfSpace=0;
  for (b=0; b<in->header.numberOfBones; b++)
  {
      for (w=0; w<in->bones[b].info->boneWeightsNumber; w++)
      {
         unsigned int boneIndex = in->bones[b].weightIndex[w];
         struct TRI_Bones_Per_Vertex_Vertice_Item * bone =  &out->bonesPerVertex[boneIndex];


          if (bone->bonesOfthisVertex < MAX_BONES_PER_VERTICE)
          {
            bone->indicesOfThisVertex[bone->bonesOfthisVertex] = in->bones[b].weightIndex[w];
            bone->weightsOfThisVertex[bone->bonesOfthisVertex] = in->bones[b].weightValue[w];
            bone->boneIDOfThisVertex[bone->bonesOfthisVertex]  = b;

            ++bone->bonesOfthisVertex;
          } else
          {
            ++outOfSpace;
          }
      }
  }

  if (outOfSpace>0)
  {
    fprintf(stderr,"Vertices are set up to accomodate at most %d bones , %u vertices where too small for our input .. \n",MAX_BONES_PER_VERTICE,outOfSpace);
  }
 return out;
}

void freeTransformTRIBonesToVertexBoneFormat(struct TRI_Bones_Per_Vertex * in)
 {
   if (in!=0)
   {
     if (in->bonesPerVertex!=0)
     {
      free(in->bonesPerVertex);
     }
    free(in);
   }
 }
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
unsigned int * convertTRIBonesToParentList(struct TRI_Model * in , unsigned int * outputNumberOfBones)
{

  *outputNumberOfBones =  in->header.numberOfBones;
  unsigned int * parentNode = ( unsigned int * )  malloc (sizeof(unsigned int) * in->header.numberOfBones);

if (parentNode!=0)
{
   memset(parentNode,0,sizeof(unsigned int) * in->header.numberOfBones);

   unsigned int i=0;
   for (i=0; i< in->header.numberOfBones; i++)
   {
     parentNode[i] = in->bones[i].info->boneParent;
   }
}
 return parentNode;
}

//Please note that the output is in the coordinate space of the binding pose model and needs to be transformed/projected etc
//according to the real location of the mesh
float * convertTRIBonesToJointPositions(struct TRI_Model * in , unsigned int * outputNumberOfJoints)
{
  if (in==0)                   { return 0; }
  if (outputNumberOfJoints==0) { return 0; }

  float * outputJoints = ( float * )  malloc (sizeof(float) * in->header.numberOfBones * 3);
  if(outputJoints!=0)
  {
  memset(outputJoints,0,sizeof(float) * in->header.numberOfBones * 3);

  unsigned int * outputNumberSamples = ( unsigned int * )  malloc (sizeof(unsigned int ) * in->header.numberOfBones);
  if (outputNumberSamples==0)
  {
    fprintf(stderr,"Could not allocate internal sample space , convertTRIBonesToJointPositions will fail \n");
    free(outputJoints);
    *outputNumberOfJoints=0;
    return 0;
  }
  memset(outputNumberSamples,0,sizeof(unsigned int) * in->header.numberOfBones);

  *outputNumberOfJoints =  in->header.numberOfBones;

  struct TRI_Bones_Per_Vertex * bpv = allocTransformTRIBonesToVertexBoneFormat(in);
  if (bpv!=0)
  {
   unsigned int i=0;
   for (i=0; i<in->header.numberOfVertices; i++)
    {
      struct TRI_Bones_Per_Vertex_Vertice_Item * bone =  &bpv->bonesPerVertex[i];

      float maxWeight = 0.0;
      unsigned int z=0 , b=0;
      for (z=0; z<bone->bonesOfthisVertex; z++)
      {
        if ( bone->weightsOfThisVertex[z]>maxWeight )
            {
              maxWeight=bone->weightsOfThisVertex[z];
              b=z;
            }
      }

      unsigned int indxID=bone->indicesOfThisVertex[b];
      unsigned int boneID=bone->boneIDOfThisVertex[b];

      if (boneID>in->header.numberOfVertices)
      {
         fprintf(stderr,"Error bug detected \n"); boneID=0;
      }

      ++outputNumberSamples[boneID];
      outputJoints[boneID*3+0]+=in->vertices[indxID*3+0];
      outputJoints[boneID*3+1]+=in->vertices[indxID*3+1];
      outputJoints[boneID*3+2]+=in->vertices[indxID*3+2];
   }


   for (i=0; i< in->header.numberOfBones; i++)
   {
     if (outputNumberSamples[i]>0)
     {
      outputJoints[i*3+0]=outputJoints[i*3+0]/outputNumberSamples[i];
      outputJoints[i*3+1]=outputJoints[i*3+1]/outputNumberSamples[i];
      outputJoints[i*3+2]=outputJoints[i*3+2]/outputNumberSamples[i];

      //fprintf(stderr,"Bone %u (%s) = ",i,in->bones[i].boneName);
      //fprintf(stderr," %0.2f,%0.2f,%0.2f  ( %u samples ) \n ",outputJoints[i*3+0],outputJoints[i*3+1],outputJoints[i*3+2],outputNumberSamples[i]);
     } else
     {
      //fprintf(stderr,"Bone %u (%s) has no samples.. \n",i,in->bones[i].boneName);
     }
   }


   freeTransformTRIBonesToVertexBoneFormat(bpv);
  }

 free(outputNumberSamples);

 }
 return outputJoints;
}

unsigned int  * getClosestVertexToJointPosition(struct TRI_Model * in , float * joints , unsigned int numberOfJoints)
{
  fprintf(stderr,"getClosestVertexToJointPosition \n");
  unsigned int * outputPositions = ( unsigned int * )  malloc (sizeof(unsigned int ) * numberOfJoints );

  if (outputPositions!=0)
  {
   memset(outputPositions,0,sizeof(unsigned int ) * numberOfJoints);

   unsigned int i=0,v=0;
   float x,y,z,diffX,diffY,diffZ , bestDistance=10000000000 , worstDistance=0 , currentDistance;

   fprintf(stderr,"searching best vertices for %u joints \n",numberOfJoints);
   for (i=0; i<numberOfJoints; i++)
   {
     x=joints[i*3+0];
     y=joints[i*3+1];
     z=joints[i*3+2];

     if ( (x!=x) || (y!=y) || (z!=z) )
     {
       //IGNORE NAN VALUE
     } else
     {
     bestDistance=10000000000;
     worstDistance=0;
     fprintf(stderr,"now doing vertice search among %u vertices \n",in->header.numberOfVertices/3);
     for (v=0; v<in->header.numberOfVertices/3; v++)
     {
       diffX=x-in->vertices[v*3+0];
       diffY=y-in->vertices[v*3+1];
       diffZ=z-in->vertices[v*3+2];

       currentDistance = (diffX*diffX) + (diffY*diffY) + (diffZ*diffZ);
       if ( currentDistance < bestDistance )
       {
         outputPositions[i]=v;
         bestDistance=currentDistance;
       }
       if ( currentDistance > worstDistance )
       {
         worstDistance=currentDistance ;
       }

       if (bestDistance==0)  //Cant find a better match..
          {break;}
     }

     fprintf(stderr,"Bone %u (%s) = ",i,in->bones[i].boneName);
     fprintf(stderr," %0.2f,%0.2f,%0.2f \n ",joints[i*3+0],joints[i*3+1],joints[i*3+2]);
     fprintf(stderr," Best Vertice is %u with a distance of %0.2f ( worst %0.2f ) \n ",outputPositions[i],sqrt(bestDistance),sqrt(worstDistance));
     }
   }


  }
 return outputPositions;
}

int setTRIModelBoneInitialPosition(struct TRI_Model * in)
{
   unsigned int outputNumberOfJoints;
   float * pos = convertTRIBonesToJointPositions( in , &outputNumberOfJoints);
   if (pos!=0)
   {
    unsigned int i=0;
    for (i=0; i<in->header.numberOfBones; i++)
    {
     in->bones[i].info->x = pos[i*3+0];
     in->bones[i].info->y = pos[i*3+1];
     in->bones[i].info->z = pos[i*3+2];

     fprintf(stderr,"Bone %u (%s) = ",i,in->bones[i].boneName);
     fprintf(stderr," %0.2f,%0.2f,%0.2f \n ",pos[i*3+0],pos[i*3+1],pos[i*3+2]);
    }

    free(pos);
    return 1;
   }

   return 0;
}
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
void colorCodeBones(struct TRI_Model * in)
{
   if (in->bones==0)
   {
     fprintf(stderr,"No bones to colorcode \n");
     return;
   }

  struct TRI_Bones_Per_Vertex * bpv = allocTransformTRIBonesToVertexBoneFormat(in);
  if (bpv!=0)
  {
   float * gp = generatePalette(in);
   if (gp!=0)
   {
    if (in->colors==0)
    {
      in->header.numberOfColors = in->header.numberOfVertices;
      in->colors = (float*) malloc(sizeof(float) * in->header.numberOfColors );
    }

   if  (
        (in->colors!=0) &&
        (in->header.numberOfColors!=0) &&
        (in->header.numberOfVertices!=0)
       )
   {
   unsigned int i=0;
   for (i=0; i<in->header.numberOfVertices; i++)
    {
      struct TRI_Bones_Per_Vertex_Vertice_Item * bone =  &bpv->bonesPerVertex[i];

      float maxWeight = 0.0;
      unsigned int z=0 , b=0;
      for (z=0; z<bone->bonesOfthisVertex; z++)
      {
        if ( bone->weightsOfThisVertex[z]>maxWeight )
            {
              maxWeight=bone->weightsOfThisVertex[z];
              b=z;
            }
      }

      unsigned int indxID=bone->indicesOfThisVertex[b];
      unsigned int boneID=bone->boneIDOfThisVertex[b];
      if (boneID>in->header.numberOfVertices)
      {
         fprintf(stderr,"Error bug detected \n"); boneID=0;
      }

      in->colors[indxID*3+0]=gp[boneID*3+0];
      in->colors[indxID*3+1]=gp[boneID*3+1];
      in->colors[indxID*3+2]=gp[boneID*3+2];
    }
   }
   free(gp);
   }
   freeTransformTRIBonesToVertexBoneFormat(bpv);
  }
}


/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
int tri_colorCodeTexture(struct TRI_Model * in, unsigned int x, unsigned int y, unsigned int width,unsigned int height)
{
  if ( (width==0) || (height==0) ) { return 0; }

  unsigned int colorStep = (255*255*255) / (width * height);
  unsigned int currentColor = colorStep;
  //------------------------------------
  if (in!=0)
  {
    if (in->textureData!=0)
    {
      unsigned int imageWidth  = in->header.textureDataWidth;
      //unsigned int imageHeight = in->header.textureDataHeight;

      unsigned int x1 = x;
      unsigned int y1 = y;
      unsigned int x2 = x + width;
      unsigned int y2 = y + height;

      unsigned char * ptr = in->textureData + (y1 * imageWidth * 3) + (x1 * 3);
      unsigned char * lineStart = ptr;
      unsigned char * lineEnd   = ptr + (width * 3);
      unsigned char * imageEnd  = in->textureData + (y2 * imageWidth * 3) + (x2 * 3);

      while (ptr<imageEnd)
      {
       while (ptr<lineEnd)
       {
        unsigned int thisPixel = currentColor;
        //------------------------------------
        unsigned char c1 = (unsigned char) thisPixel % 255;
        thisPixel = thisPixel / 255;
        unsigned char c2 = (unsigned char) thisPixel % 255;
        thisPixel = thisPixel / 255;
        unsigned char c3 = (unsigned char) thisPixel % 255;
        //------------------------------------
        *ptr = c3;  ++ptr; // R
        *ptr = c2;  ++ptr; // G
        *ptr = c1;  ++ptr; // B
        //---------------------------
        currentColor += colorStep;
       }
       //--------------------------
       lineStart += imageWidth * 3;
       lineEnd   += imageWidth * 3;
       ptr = lineStart;
       //--------------------------
      }

      return 1;
    }
  }
  return 0;
}
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------


int setTRIJointRotationOrder(
                             struct TRI_Model * in ,
                             unsigned int jointToChange ,
                             unsigned int rotationOrder
                           )
{
  if (in==0)        { return 0; }
  if (in->bones==0) { return 0; }

  if ( jointToChange < in->header.numberOfBones )
  {
    fprintf(stderr,"setTRIJointRotationOrder : joint=%u -> order=%u ",jointToChange,rotationOrder);
    if (rotationOrder<ROTATION_ORDER_NUMBER_OF_NAMES)
    {
       fprintf(stderr," %s \n",ROTATION_ORDER_NAMESA[rotationOrder]);
    } else
    {
       fprintf(stderr,"\n");
    }

    in->bones[jointToChange].info->eulerRotationOrder = (unsigned char) rotationOrder;
    return 1;
  }
 return 0;
}

int getTRIJointRotationOrder(
                              struct TRI_Model * in ,
                              unsigned int jointToChange ,
                              unsigned int rotationOrder
                            )
{
  if (in==0)        { return 0; }
  if (in->bones==0) { return 0; }

  unsigned char val =  in->bones[jointToChange].info->eulerRotationOrder;

  return (int) val;
}

void transformTRIJoint(
                        struct TRI_Model * in ,
                        float * jointData ,
                        unsigned int jointDataSize ,

                        unsigned int jointToChange ,
                        float rotEulerX , //Roll
                        float rotEulerY , //Pitch
                        float rotEulerZ   //Yaw
                      )
{
  //This is needed for meta reasons
  in->bones[jointToChange].info->rotX = rotEulerX;
  in->bones[jointToChange].info->rotY = rotEulerY;
  in->bones[jointToChange].info->rotZ = rotEulerZ;

  //We set the 4x4 Matrix that is what is used for the transform..
  float * mat = &jointData[16*jointToChange];

  _triTrans_create4x4MatrixFromEulerAnglesZYX(mat,rotEulerX,rotEulerY,rotEulerZ);
}

float * mallocModelTransformJoints(
                                    struct TRI_Model * triModelInput ,
                                    unsigned int * jointDataSizeOutput
                                   )
{
  float * returnMat = (float * ) malloc(sizeof(float) * 16 * triModelInput->header.numberOfBones);
  if (returnMat)
  {
     *jointDataSizeOutput =  triModelInput->header.numberOfBones;
     unsigned int i=0;
     for (i=0; i<(*jointDataSizeOutput); i++)
     {
       float * m = &returnMat[16*i];
       create4x4FIdentityMatrixDirect(m);
       //m[0] = 1.0;  m[1] = 0.0;  m[2] = 0.0;   m[3] = 0.0;
       //m[4] = 0.0;  m[5] = 1.0;  m[6] = 0.0;   m[7] = 0.0;
       //m[8] = 0.0;  m[9] = 0.0;  m[10] = 1.0;  m[11] =0.0;
       //m[12]= 0.0;  m[13]= 0.0;  m[14] = 0.0;  m[15] = 1.0;
     }
  }
  return returnMat;
}

float * mallocModelTransformJointsEulerAnglesDegrees(
                                                      struct TRI_Model * triModelInput ,
                                                      float * jointData ,
                                                      unsigned int jointDataSize ,
                                                      unsigned int method
                                                     )
{
 float * returnMat = (float * ) malloc(sizeof(float) * 3 * triModelInput->header.numberOfBones);
 if (returnMat)
  {
     float euler[4]={0};
     float quaternions[4]={0};
     float m4x4[16]={0};

     unsigned int i=0;
     for (i=0; i<jointDataSize; i++)
     {
       float * mat = &jointData[16*i];

       copy4x4FMatrix(m4x4,mat);
       matrix4x42Quaternion(quaternions,qXqYqZqW,m4x4);

       quaternions2Euler(euler,quaternions,qXqYqZqW);

       returnMat[i*3+0] = euler[0];
       returnMat[i*3+1] = euler[1];
       returnMat[i*3+2] = euler[2];
     }
  }
  return returnMat;
}

void printModelTransform(struct TRI_Model * in)
{
  for (unsigned int i=0; i<in->header.numberOfBones; i++)
    {
      if (in->bones[i].info->altered)
      {
        fprintf(stderr,"POSE4x4(this,0,%s",in->bones[i].boneName);

        for (unsigned int z=0; z<16; z++)
        {
          fprintf(stderr,",%0.3f",in->bones[i].info->finalVertexTransformation[z]);
        }
        fprintf(stderr,")\n");
      }
    }
}

/* This is direct setting of the joint data , overwriting default values */
void recursiveJointHierarchyTransformer(
                                        struct TRI_Model * in  ,
                                        int curBone ,
                                        struct Matrix4x4OfFloats parentTransformUntouched,
                                        float * joint4x4Data , unsigned int joint4x4DataSize ,
                                        unsigned int recursionLevel
                                       )
{
  //Sanity check..
  //-----------------------------
  if (in==0) { return; }
  //-----------------------------
  if (recursionLevel>=in->header.numberOfBones+1)
        { fprintf(stderr,RED "BUG : REACHED RECURSION LIMIT (%u/%u)\n" NORMAL,recursionLevel,in->header.numberOfBones); return; }
  //-----------------------------

  struct Matrix4x4OfFloats globalTransformation={0};
  struct Matrix4x4OfFloats nodeTransformation  ={0};
  struct Matrix4x4OfFloats parentTransform     ={0};
  copy4x4FMatrixToAlignedContainer(&parentTransform,parentTransformUntouched.m);


  //We use nodeLocalTransformation as shorthand so that we don't have to access the bone structure every time
  struct Matrix4x4OfFloats nodeLocalTransformation;
  copy4x4FMatrixToAlignedContainer(&nodeLocalTransformation,in->bones[curBone].info->localTransformation);

  if ( (joint4x4Data!=0) && ((unsigned int)curBone*16<joint4x4DataSize) )
     {
       struct Matrix4x4OfFloats joint4x4DataPacked   ={0};
       copy4x4FMatrixToAlignedContainer(&joint4x4DataPacked,&joint4x4Data[curBone*16]);
       //We do the transformation of our node with the new joint 4x4 data we received..!  nodeTransformationCopy
       multiplyTwo4x4FMatricesS(&nodeTransformation,&nodeLocalTransformation,&joint4x4DataPacked);

     } else
     {
       //If there is no 4x4 transform to use then just copy our local transformation
       fprintf(stderr,YELLOW "Bone %u has no joint transform.. \n" NORMAL,curBone);
       copy4x4FMatrix(nodeTransformation.m,nodeLocalTransformation.m);
     }

  //We calculate the globalTransformation of the node by chaining it to its parent..!
  multiplyTwo4x4FMatricesS(
                           &globalTransformation,
                           &parentTransform,
                           &nodeTransformation
                          );

  //Since we have everything ready, let's store the bone position..
  struct Vector4x1OfFloats boneCenter={0}; boneCenter.m[3]=1.0;
  transform3DPointFVectorUsing4x4FMatrix(&boneCenter,&globalTransformation,&boneCenter);
  in->bones[curBone].info->x = boneCenter.m[0];
  in->bones[curBone].info->y = boneCenter.m[1];
  in->bones[curBone].info->z = boneCenter.m[2];


  //We calculate the finalVertexTransformation for all vertices that are influenced for this bone
  //by chaining the global transformation with the bone's global inverse transform and and rest/bind pose transform
  //Apply the rotation matrix on top of the default one (inverse rot of the matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose)
  struct Matrix4x4OfFloats finalVertexTransformation;
  struct Matrix4x4OfFloats boneGlobalInverseTransform;
  struct Matrix4x4OfFloats matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose;

  copy4x4FMatrixToAlignedContainer(&matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose,in->bones[curBone].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose);
  copy4x4FMatrixToAlignedContainer(&boneGlobalInverseTransform,in->header.boneGlobalInverseTransform); //This is repeated many times for no reason

  multiplyThree4x4FMatrices(
                            &finalVertexTransformation ,
                            &boneGlobalInverseTransform ,
                            &globalTransformation,
                            &matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose
                           );

  //Copy back our calculated output to the TRI bones
  copy4x4FMatrix(in->bones[curBone].info->finalVertexTransformation,finalVertexTransformation.m);


   //Each bone might have multiple children, we recursively execute the same transform for all children of this node..!
  for (unsigned int childID = 0; childID < in->bones[curBone].info->numberOfBoneChildren; childID++)
      {
        recursiveJointHierarchyTransformer(
                                            in,
                                            in->bones[curBone].boneChild[childID], //recursively execute on this bone's child
                                            globalTransformation,
                                            joint4x4Data,joint4x4DataSize,
                                            recursionLevel+1
                                           );
      }

  return;
}









/* This is direct setting of the joint data , overwriting default values */
void recursiveJointHierarchyTransformerDirect(
                                               struct TRI_Model * in  ,
                                               int curBone ,
                                               float * parentTransformUntouched ,
                                               float * joint4x4Data , unsigned int joint4x4DataSize ,
                                               unsigned int recursionLevel
                                             )
{
  //Sanity check..
  //-----------------------------
  if (in==0) { return; }
  //-----------------------------
  if (recursionLevel>=in->header.numberOfBones+1)
        { fprintf(stderr,RED "BUG : REACHED RECURSION LIMIT (%u/%u)\n" NORMAL,recursionLevel,in->header.numberOfBones); return; }
  //-----------------------------

  float emptyParentTransform[16], globalTransformation[16], nodeTransformation[16];
  float * parentTransform = parentTransformUntouched;

  //int multiplyThree4x4FMatrices(struct Matrix4x4OfFloats * result,struct Matrix4x4OfFloats * matrixA,struct Matrix4x4OfFloats * matrixB,struct Matrix4x4OfFloats * matrixC);

  if (parentTransformUntouched==0)
   {
      //If parentTransformUntouched is empty then use an identity matrix locally allocated in our emptyParentTransform
      //as a parentTransform, in any case we do not touch parentTransformUntouched..!
      create4x4FIdentityMatrixDirect((float*) &emptyParentTransform);
      parentTransform = emptyParentTransform;
   }

  //We use nodeLocalTransformation as shorthand so that we don't have to access the bone structure every time
  float * nodeLocalTransformation = in->bones[curBone].info->localTransformation;

  if ( (joint4x4Data!=0) && ((unsigned int)curBone*16<joint4x4DataSize) )
     {
       //We do the transformation of our node with the new joint 4x4 data we received..!  nodeTransformationCopy
       multiplyTwoRaw4x4FMatricesS(nodeTransformation,nodeLocalTransformation,&joint4x4Data[curBone*16]);
     } else
     {
       //If there is no 4x4 transform to use then just copy our local transformation
       fprintf(stderr,YELLOW "Bone %u has no joint transform.. \n" NORMAL,curBone);
       copy4x4FMatrix(nodeTransformation,nodeLocalTransformation);
     }

  //We calculate the globalTransformation of the node by chaining it to its parent..!
  multiplyTwoRaw4x4FMatricesS(
                              (float*) globalTransformation,
                              parentTransform,
                              (float*) nodeTransformation
                             );

  //Since we have everything ready, let's store the bone position..
  struct Vector4x1OfFloats boneCenter={0}; boneCenter.m[3]=1.0;
  transform3DPointFVectorUsing4x4FMatrix_Naive(boneCenter.m,globalTransformation,boneCenter.m);
  in->bones[curBone].info->x = boneCenter.m[0];
  in->bones[curBone].info->y = boneCenter.m[1];
  in->bones[curBone].info->z = boneCenter.m[2];


  //We calculate the finalVertexTransformation for all vertices that are influenced for this bone
  //by chaining the global transformation with the bone's global inverse transform and and rest/bind pose transform
  //Apply the rotation matrix on top of the default one (inverse rot of the matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose)
  multiplyThree4x4FMatrices_Naive(
                                   in->bones[curBone].info->finalVertexTransformation ,
                                   in->header.boneGlobalInverseTransform ,
                                   globalTransformation,
                                   in->bones[curBone].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose
                                  );

   //Each bone might have multiple children, we recursively execute the same transform for all children of this node..!
  for (unsigned int childID = 0; childID < in->bones[curBone].info->numberOfBoneChildren; childID++)
      {
        recursiveJointHierarchyTransformerDirect(
                                                 in,
                                                 in->bones[curBone].boneChild[childID], //recursively execute on this bone's child
                                                 globalTransformation,
                                                 joint4x4Data,joint4x4DataSize,
                                                 recursionLevel+1
                                                );
      }

  return;
}


int applyVertexTransformation( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn )
{
   if ( (triModelIn->vertices==0) && (triModelIn->normal==0) )
       {
         fprintf(stderr,YELLOW "Cannot apply vertex transformation with no vertices or normals.. \n" NORMAL);
         return 0;
       }

  struct Vector4x1OfFloats transformedPosition={0},transformedNormal={0},position={0},normal={0};

  //We NEED to clear the vertices and normals since they are added uppon , not having
  //the next two lines results in really weird and undebuggable visual behaviour
  memset(triModelOut->vertices,0,triModelOut->header.numberOfVertices * sizeof(float));

  //Clean normal output before repopulating it..
  if (triModelIn->normal!=0)
       {
         memset(triModelOut->normal,0,triModelOut->header.numberOfNormals * sizeof(float));
       }

  //We will need a Matrix4x4OfFloats since it is aligned to give SSE speedups..
  struct  Matrix4x4OfFloats boneTransformMatrix;

  for (unsigned int boneID=0; boneID<triModelIn->header.numberOfBones; boneID++)
   {
     if ( is4x4FZeroMatrix(triModelIn->bones[boneID].info->finalVertexTransformation) )
     {
       fprintf(stderr,RED "Joint Transform was zero for bone %s (%u) , there was a bug preparing the matrices \n" NORMAL,triModelIn->bones[boneID].boneName , boneID );
       float * m = triModelIn->bones[boneID].info->finalVertexTransformation;
       create4x4FIdentityMatrixDirect(m);
     }

     //CPU bone transformations..!
     for (unsigned int boneWeightID=0; boneWeightID<triModelIn->bones[boneID].info->boneWeightsNumber; boneWeightID++)
     {
       //V is the vertice we will be working in this loop
       unsigned int vertexID = triModelIn->bones[boneID].weightIndex[boneWeightID];
       //W is the weight that we have for the specific bone
       float boneWeightValue = triModelIn->bones[boneID].weightValue[boneWeightID];

       //Vertice transformation ----------------------------------------------
       //We load our input into position/normal
       position.m[0] = triModelIn->vertices[vertexID*3+0];
       position.m[1] = triModelIn->vertices[vertexID*3+1];
       position.m[2] = triModelIn->vertices[vertexID*3+2];
       position.m[3] = 1.0;

       //Keep a copy of our matrix on our SSE aligned Matrix4x4OfFloats structure
       copy4x4FMatrix(boneTransformMatrix.m,triModelIn->bones[boneID].info->finalVertexTransformation);

       //We transform input (initial) position with the transform we computed to get transformedPosition
       transform3DPointFVectorUsing4x4FMatrix(&transformedPosition,&boneTransformMatrix,&position);
       //Please note that triModelOut->vertices is set to 0 by memset call above so it is clean..!
       triModelOut->vertices[vertexID*3+0] += (float) transformedPosition.m[0] * boneWeightValue;
       triModelOut->vertices[vertexID*3+1] += (float) transformedPosition.m[1] * boneWeightValue;
       triModelOut->vertices[vertexID*3+2] += (float) transformedPosition.m[2] * boneWeightValue;
       //----------------------------------------------------------------------

       //Normal transformation ----------------------------------------------
       if (triModelIn->normal!=0)
       {
        normal.m[0]   = triModelIn->normal[vertexID*3+0];
        normal.m[1]   = triModelIn->normal[vertexID*3+1];
        normal.m[2]   = triModelIn->normal[vertexID*3+2];
        normal.m[3]   = 0.0;

        //We transform input (initial) normal with the transform we computed to get transformedNormal
        transform3DNormalVectorUsing3x3FPartOf4x4FMatrix(transformedNormal.m,&boneTransformMatrix,normal.m);
        triModelOut->normal[vertexID*3+0] += (float) transformedNormal.m[0] * boneWeightValue;
        triModelOut->normal[vertexID*3+1] += (float) transformedNormal.m[1] * boneWeightValue;
        triModelOut->normal[vertexID*3+2] += (float) transformedNormal.m[2] * boneWeightValue;
       }
       //----------------------------------------------------------------------
     }
   }
 return 1;
}








int doModelTransform(
                      struct TRI_Model * triModelOut,
                      struct TRI_Model * triModelIn,
                      float * joint4x4Data,
                      unsigned int joint4x4DataSize ,
                      unsigned int autodetectAlteredMatrices,//This is no longer used
                      unsigned int directSettingOfMatrices, //This is no longer used
                      unsigned int performVertexTransform, //If you want to handle the transform on a shader set this to 0
                      unsigned int jointAxisConvention
                    )
{
 if (triModelIn==0)
                     { fprintf(stderr,"doModelTransform called without input TRI Model \n"); return 0; }

 if ( ( triModelIn->vertices ==0 ) || ( triModelIn->header.numberOfVertices ==0 ) )
                     { fprintf(stderr,RED "Number of vertices is zero so can't do model transform using weights..\n" NORMAL); return 0; }

 if ( (joint4x4Data==0) || (joint4x4DataSize==0) )
 {
   fprintf(stderr,"doModelTransform called without joints to transform , ");
   fprintf(stderr,"so it will be just returning a null transformed copy of");
   fprintf(stderr,"the input mesh , hope this is what you intended..\n");
   return 0;
 }

 for (unsigned int boneID=0; boneID<triModelIn->header.numberOfBones; boneID++)
   {
     //All matrices considered and marked altered when calling this call
     triModelIn->bones[boneID].info->altered=1;
   }

  //Use simple transformer that does not rely on struct Matrix4x4OfFloats
  #define USE_SIMPLE_VERSION 0

  #if USE_SIMPLE_VERSION
  float initialParentTransform[16]={0};
  //The initial parent transform is an identity matrix..!
  create4x4FIdentityMatrixDirect((float*) &initialParentTransform);

  //This recursively calculates all matrix transforms and prepares the correct matrices in triModelIn
  //each boneID gets its final 4x4 matrix in triModelIn->bones[boneID].info->finalVertexTransformation
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  recursiveJointHierarchyTransformerDirect(
                                           triModelIn ,
                                           triModelIn->header.rootBone  ,
                                           initialParentTransform ,
                                           joint4x4Data ,
                                           joint4x4DataSize ,
                                           0 /*First call 0 level recursion*/
                                          );
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #else
  struct Matrix4x4OfFloats parentMatrix;
  create4x4FIdentityMatrix(&parentMatrix);

  //This recursively calculates all matrix transforms and prepares the correct matrices in triModelIn
  //each boneID gets its final 4x4 matrix in triModelIn->bones[boneID].info->finalVertexTransformation
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  recursiveJointHierarchyTransformer(
                                     triModelIn ,
                                     triModelIn->header.rootBone  ,
                                     parentMatrix ,
                                     joint4x4Data ,
                                     joint4x4DataSize ,
                                     0 /*First call 0 level recursion*/
                                    );
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #endif


  if (performVertexTransform)
  {
    //Past checks..
   tri_copyModel(triModelOut,triModelIn,1,0); //Last 1 means we also want bone data , Last 0 means we dont need to copy texture data
   applyVertexTransformation(triModelOut,triModelIn);
  }

 return 1;
}

/* ============================================================================
 * MHR Linear Blend Skinning — native C implementation
 *
 * Replicates body_model.pt forward pass:
 *   blend_shape → parameter_transform → local_skeleton → FK → LBS
 *
 * All quaternions stored XYZW throughout.
 * ============================================================================ */

#define MHR_LBS_MAGIC   0x4C425300u  /* 'LBS\0' */
#define MHR_LBS_VERSION 3u  /* v3 appends hand_pose_mean[54] + hand_pose_comps[54×54]
                                + hand_joint_idxs_left[27] + hand_joint_idxs_right[27] */
#define MHR_LN2         0.693147180559945f

/* ── static quaternion / vector helpers ──────────────────────────────────── */

/* Hamilton product r = a * b  (XYZW storage: [x, y, z, w]).
 *
 * Rotation semantics: when you rotate a vector v with quaternion q,
 * the result is q*v*q^{-1}.  For a composed quaternion r = a*b:
 *   r*v*r^{-1} = a*(b*v*b^{-1})*a^{-1}
 * meaning b's rotation is applied FIRST to v, then a's.
 * Equivalently, the rotation matrix for r is R_a @ R_b.
 *
 * Usage patterns:
 *   mhr_qmul(g_q, parent_q, child_q)  →  R_global = R_parent @ R_child  (FK)
 *   mhr_qmul(q_local, pre_q, euler_q) →  R_local  = R_pre    @ R_euler  (Step 3)
 *   mhr_qmul(skin_q,  g_q,   ib_q)   →  R_skin   = R_global @ R_inv_bind (Step 5)
 */
static void mhr_qmul(float *r, const float *a, const float *b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}

/* Rotate vector v by unit quaternion q (XYZW): computes q*v*q^{-1}. */
static void mhr_qrot(float *out, const float *q, const float *v)
{
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3];
    float vx=v[0], vy=v[1], vz=v[2];
    float tx = 2.f*(qy*vz - qz*vy);
    float ty = 2.f*(qz*vx - qx*vz);
    float tz = 2.f*(qx*vy - qy*vx);
    out[0] = vx + qw*tx + (qy*tz - qz*ty);
    out[1] = vy + qw*ty + (qz*tx - qx*tz);
    out[2] = vz + qw*tz + (qx*ty - qy*tx);
}

/* Intrinsic XYZ Euler angles (radians) → unit quaternion (XYZW).
 *
 * PyMomentum local_skeleton convention: given joint euler params [ex, ey, ez],
 * the joint rotation matrix is R = Rz(ez) * Ry(ey) * Rx(ex)  — i.e., X applied
 * first to vectors, then Y, then Z.  This is "intrinsic XYZ" = "extrinsic ZYX".
 * See pymomentum/trs.py  rotmat_from_euler_xyz("xyz", angles).
 *
 * Quaternion form: q = qz * qy * qx
 *   because mhr_qmul(r,a,b) gives R_a @ R_b, so
 *   mhr_qmul(tmp, qz, qy)  → R_tmp = R_z @ R_y
 *   mhr_qmul(q,   tmp, qx) → R_q   = R_z @ R_y @ R_x  ✓
 *
 * The Euler angles (ex,ey,ez) arriving here are extracted by rot6d_to_euler
 * (preprocess.hpp) using the ZYX decomposition: ex=rx, ey=ry, ez=rz such that
 * R_original = Rz(rz)*Ry(ry)*Rx(rx).  They match what PyMomentum stores in
 * joint_params[3:6] after the parameter transform.
 */
static void mhr_euler_xyz_to_quat(float ex, float ey, float ez, float *q)
{
    float hx=ex*0.5f, hy=ey*0.5f, hz=ez*0.5f;
    float qx[4]={sinf(hx),0.f,     0.f,     cosf(hx)};
    float qy[4]={0.f,     sinf(hy),0.f,     cosf(hy)};
    float qz[4]={0.f,     0.f,     sinf(hz),cosf(hz)};
    float tmp[4];
    mhr_qmul(tmp, qz, qy);   /* R_tmp = Rz @ Ry */
    mhr_qmul(q,   tmp, qx);  /* R_q   = Rz @ Ry @ Rx  (PyMomentum XYZ intrinsic) */
}

/* ── loader / free ────────────────────────────────────────────────────────── */

struct MHR_LBS_Data *mhr_lbs_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr,"mhr_lbs_load: cannot open %s\n", path); return NULL; }

    unsigned int hdr[8];
    if (fread(hdr, sizeof(unsigned int), 8, f) != 8) {
        fprintf(stderr,"mhr_lbs_load: short header\n"); fclose(f); return NULL;
    }
    if (hdr[0] != MHR_LBS_MAGIC || (hdr[1] != 1u && hdr[1] != 2u && hdr[1] != MHR_LBS_VERSION)) {
        fprintf(stderr,"mhr_lbs_load: bad magic/version (got %u, expected 1, 2 or %u)\n",
                hdr[1], MHR_LBS_VERSION);
        fclose(f); return NULL;
    }
    unsigned int file_version = hdr[1];

    struct MHR_LBS_Data *d = (struct MHR_LBS_Data*)calloc(1, sizeof(*d));
    if (!d) { fclose(f); return NULL; }

    d->n_joints   = (int)hdr[2];
    d->n_skin     = (int)hdr[3];
    d->n_verts    = (int)hdr[4];
    d->n_shape_pc = (int)hdr[5];
    d->n_face_pc  = (int)hdr[6];
    d->pt_cols    = (int)hdr[7];
    d->pt_rows    = d->n_joints * 7;

    fprintf(stderr,"mhr_lbs: joints=%d skin=%d verts=%d shape_pc=%d face_pc=%d\n",
            d->n_joints, d->n_skin, d->n_verts, d->n_shape_pc, d->n_face_pc);

#define LBS_ALLOC_F(field, n) \
    d->field = (float*)malloc((size_t)(n)*sizeof(float)); if (!d->field) goto oom;
#define LBS_ALLOC_I(field, n) \
    d->field = (int*)malloc((size_t)(n)*sizeof(int));   if (!d->field) goto oom;
#define LBS_READ_F(field, n) \
    if (fread(d->field, sizeof(float), (size_t)(n), f) != (size_t)(n)) goto io_err;
#define LBS_READ_I(field, n) \
    if (fread(d->field, sizeof(int),   (size_t)(n), f) != (size_t)(n)) goto io_err;

    LBS_ALLOC_F(PT,                 (size_t)d->pt_rows * d->pt_cols)
    LBS_ALLOC_F(joint_offsets,      d->n_joints * 3)
    LBS_ALLOC_F(joint_prerotations, d->n_joints * 4)
    LBS_ALLOC_I(joint_parents,      d->n_joints)
    LBS_ALLOC_F(inv_bind_pose,      d->n_joints * 8)
    LBS_ALLOC_I(skin_joint_idx,     d->n_skin)
    LBS_ALLOC_F(skin_weights,       d->n_skin)
    LBS_ALLOC_I(skin_vert_idx,      d->n_skin)
    LBS_ALLOC_F(base_shape,         d->n_verts * 3)
    LBS_ALLOC_F(shape_vectors,      (size_t)d->n_shape_pc * d->n_verts * 3)
    LBS_ALLOC_F(face_vectors,       (size_t)d->n_face_pc  * d->n_verts * 3)

    LBS_READ_F(PT,                 (size_t)d->pt_rows * d->pt_cols)
    LBS_READ_F(joint_offsets,      d->n_joints * 3)
    LBS_READ_F(joint_prerotations, d->n_joints * 4)
    LBS_READ_I(joint_parents,      d->n_joints)
    LBS_READ_F(inv_bind_pose,      d->n_joints * 8)
    LBS_READ_I(skin_joint_idx,     d->n_skin)
    LBS_READ_F(skin_weights,       d->n_skin)
    LBS_READ_I(skin_vert_idx,      d->n_skin)
    LBS_READ_F(base_shape,         d->n_verts * 3)
    LBS_READ_F(shape_vectors,      (size_t)d->n_shape_pc * d->n_verts * 3)
    LBS_READ_F(face_vectors,       (size_t)d->n_face_pc  * d->n_verts * 3)

    /* version 2: scale PCA data (scale_mean [68], scale_comps [28×68]) */
    if (file_version >= 2u) {
        d->n_scale_pc  = 28;
        d->n_scale_out = 68;
        LBS_ALLOC_F(scale_mean,  d->n_scale_out)
        LBS_ALLOC_F(scale_comps, d->n_scale_pc * d->n_scale_out)
        LBS_READ_F(scale_mean,  d->n_scale_out)
        LBS_READ_F(scale_comps, d->n_scale_pc * d->n_scale_out)
        fprintf(stderr,"mhr_lbs: loaded scale PCA (mean[%d], comps[%dx%d])\n",
                d->n_scale_out, d->n_scale_pc, d->n_scale_out);
    }

    /* version 3: hand pose PCA + per-hand joint indices */
    if (file_version >= 3u) {
        d->n_hand_pca = 54;
        d->n_hand_out = 27;
        LBS_ALLOC_F(hand_pose_mean,  d->n_hand_pca)
        LBS_ALLOC_F(hand_pose_comps, d->n_hand_pca * d->n_hand_pca)
        LBS_ALLOC_I(hand_joint_idxs_left,  d->n_hand_out)
        LBS_ALLOC_I(hand_joint_idxs_right, d->n_hand_out)
        LBS_READ_F(hand_pose_mean,  d->n_hand_pca)
        LBS_READ_F(hand_pose_comps, d->n_hand_pca * d->n_hand_pca)
        LBS_READ_I(hand_joint_idxs_left,  d->n_hand_out)
        LBS_READ_I(hand_joint_idxs_right, d->n_hand_out)
        fprintf(stderr,"mhr_lbs: loaded hand PCA (mean[%d], comps[%dx%d]) + joint idxs L/R[%d]\n",
                d->n_hand_pca, d->n_hand_pca, d->n_hand_pca, d->n_hand_out);
    }

#undef LBS_ALLOC_F
#undef LBS_ALLOC_I
#undef LBS_READ_F
#undef LBS_READ_I

    fclose(f);
    fprintf(stderr,"mhr_lbs: loaded %s OK (version %u)\n", path, file_version);
    return d;

oom:
    fprintf(stderr,"mhr_lbs_load: out of memory\n");
    fclose(f); mhr_lbs_free(d); return NULL;
io_err:
    fprintf(stderr,"mhr_lbs_load: short read on %s\n", path);
    fclose(f); mhr_lbs_free(d); return NULL;
}

void mhr_lbs_free(struct MHR_LBS_Data *d)
{
    if (!d) return;
    free(d->PT);               free(d->joint_offsets);
    free(d->joint_prerotations); free(d->joint_parents);
    free(d->inv_bind_pose);    free(d->skin_joint_idx);
    free(d->skin_weights);     free(d->skin_vert_idx);
    free(d->base_shape);       free(d->shape_vectors);
    free(d->face_vectors);     free(d->scale_mean);
    free(d->scale_comps);
    free(d->hand_pose_mean);   free(d->hand_pose_comps);
    free(d->hand_joint_idxs_left); free(d->hand_joint_idxs_right);
    free(d->corr_sp1_row); free(d->corr_sp1_col); free(d->corr_sp1_val);
    free(d->corr_sp2_row); free(d->corr_sp2_col); free(d->corr_sp2_val);
    free(d);
}

/* ── correctives loader ──────────────────────────────────────────────────────
 *
 * correctives.bin format (all little-endian):
 *   uint32  magic    = 0x43524543
 *   uint32  version  = 1
 *   int32   n_joints, n_verts, n_feat, n_hidden, n_out, nnz1, nnz2
 *   int32   sp1_row[nnz1], sp1_col[nnz1], float sp1_val[nnz1]
 *   int32   sp2_row[nnz2], sp2_col[nnz2], float sp2_val[nnz2]
 */
#define MHR_CORR_MAGIC 0x43524543u

int mhr_correctives_load(struct MHR_LBS_Data *d, const char *path)
{
    if (!d || !path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[corr] correctives.bin not found: %s\n", path);
        return 0;
    }

    unsigned int magic, ver;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        magic != MHR_CORR_MAGIC || ver != 1u) {
        fprintf(stderr, "[corr] bad magic/version in %s\n", path);
        fclose(f); return 0;
    }

    int nj, nv, nfeat, nhid, nout, nnz1, nnz2;
    int hdr[7];
    if (fread(hdr, 4, 7, f) != 7) { fclose(f); return 0; }
    nj = hdr[0]; nv = hdr[1]; nfeat = hdr[2];
    nhid = hdr[3]; nout = hdr[4]; nnz1 = hdr[5]; nnz2 = hdr[6];

    if (nj != d->n_joints || nv != d->n_verts) {
        fprintf(stderr, "[corr] mismatch: file says nj=%d nv=%d, lbs has nj=%d nv=%d\n",
                nj, nv, d->n_joints, d->n_verts);
        fclose(f); return 0;
    }

#define CORR_ALLOC_I(field, n) \
    d->field = (int*)malloc((size_t)(n)*sizeof(int)); if (!d->field) goto oom;
#define CORR_ALLOC_F(field, n) \
    d->field = (float*)malloc((size_t)(n)*sizeof(float)); if (!d->field) goto oom;

    CORR_ALLOC_I(corr_sp1_row, nnz1) CORR_ALLOC_I(corr_sp1_col, nnz1)
    CORR_ALLOC_F(corr_sp1_val, nnz1)
    CORR_ALLOC_I(corr_sp2_row, nnz2) CORR_ALLOC_I(corr_sp2_col, nnz2)
    CORR_ALLOC_F(corr_sp2_val, nnz2)
#undef CORR_ALLOC_I
#undef CORR_ALLOC_F

    if (fread(d->corr_sp1_row, 4, nnz1, f) != (size_t)nnz1) goto read_err;
    if (fread(d->corr_sp1_col, 4, nnz1, f) != (size_t)nnz1) goto read_err;
    if (fread(d->corr_sp1_val, 4, nnz1, f) != (size_t)nnz1) goto read_err;
    if (fread(d->corr_sp2_row, 4, nnz2, f) != (size_t)nnz2) goto read_err;
    if (fread(d->corr_sp2_col, 4, nnz2, f) != (size_t)nnz2) goto read_err;
    if (fread(d->corr_sp2_val, 4, nnz2, f) != (size_t)nnz2) goto read_err;

    d->corr_n_feat   = nfeat;
    d->corr_n_hidden = nhid;
    d->corr_n_out    = nout;
    d->corr_nnz1     = nnz1;
    d->corr_nnz2     = nnz2;
    fclose(f);
    fprintf(stderr, "[corr] loaded %s  nnz1=%d nnz2=%d\n", path, nnz1, nnz2);
    return 1;

read_err:
    fprintf(stderr, "[corr] read error in %s\n", path);
oom:
    free(d->corr_sp1_row); free(d->corr_sp1_col); free(d->corr_sp1_val);
    free(d->corr_sp2_row); free(d->corr_sp2_col); free(d->corr_sp2_val);
    d->corr_sp1_row = d->corr_sp1_col = NULL; d->corr_sp1_val = NULL;
    d->corr_sp2_row = d->corr_sp2_col = NULL; d->corr_sp2_val = NULL;
    fclose(f); return 0;
}

/* Compute pose-corrective 6D features from joint_params.
 * joint_params: [n_joints*7] (output of parameter transform)
 * Joints 2..n_joints-1 contribute; joints 0,1 are skipped (matches Python).
 * out_feat: [(n_joints-2)*6]
 * R = Rz(ez) @ Ry(ey) @ Rx(ex), 6D = first 2 columns, then subtract identity. */
static void mhr_compute_pose_features(const float *joint_params, int n_joints, float *out_feat)
{
    int feat_idx = 0;
    for (int j = 2; j < n_joints; j++) {
        float ex = joint_params[j*7 + 3];  /* rx */
        float ey = joint_params[j*7 + 4];  /* ry */
        float ez = joint_params[j*7 + 5];  /* rz */
        float cx = cosf(ex), sx = sinf(ex);
        float cy = cosf(ey), sy = sinf(ey);
        float cz = cosf(ez), sz = sinf(ez);
        /* col0 of R = Rz @ Ry @ Rx */
        out_feat[feat_idx + 0] = cz*cy       - 1.f;  /* R[0,0] − 1 */
        out_feat[feat_idx + 1] = sz*cy;               /* R[1,0]     */
        out_feat[feat_idx + 2] = -sy;                  /* R[2,0]     */
        /* col1 of R */
        out_feat[feat_idx + 3] = -sz*cx + cz*sy*sx;  /* R[0,1]     */
        out_feat[feat_idx + 4] = cz*cx + sz*sy*sx - 1.f; /* R[1,1] − 1 */
        out_feat[feat_idx + 5] = cy*sx;               /* R[2,1]     */
        feat_idx += 6;
    }
}

/* Sparse matrix-vector product: y += A(sparse COO) * x, then clip negatives (ReLU).
 * Accumulates into y (caller must zero-init). */
static void mhr_spmv_relu(const int *rows, const int *cols, const float *vals,
                           int nnz, const float *x, float *y)
{
    for (int k = 0; k < nnz; k++)
        y[rows[k]] += vals[k] * x[cols[k]];
    /* ReLU is applied outside since layer2 does NOT use it */
}

/* Apply pose correctives: adds corrective vertex offsets to unposed[nv*3] in-place.
 * Returns 0 if correctives not loaded (no-op). */
static int mhr_apply_correctives(const struct MHR_LBS_Data *d,
                                  const float *joint_params,
                                  float *unposed)
{
    if (!d->corr_sp1_row) return 0;  /* not loaded */

    int nfeat  = d->corr_n_feat;
    int nhid   = d->corr_n_hidden;
    int nout   = d->corr_n_out;
    int nnz1   = d->corr_nnz1;
    int nnz2   = d->corr_nnz2;

    float *feat   = (float*)calloc((size_t)nfeat, sizeof(float));
    float *hidden = (float*)calloc((size_t)nhid,  sizeof(float));
    float *out    = (float*)calloc((size_t)nout,  sizeof(float));
    if (!feat || !hidden || !out) { free(feat); free(hidden); free(out); return 0; }

    /* Feature extraction: euler angles of joints 2..N-1 → 6D centered */
    mhr_compute_pose_features(joint_params, d->n_joints, feat);

    /* Layer 1: sparse matmul → [nhid], then ReLU */
    mhr_spmv_relu(d->corr_sp1_row, d->corr_sp1_col, d->corr_sp1_val, nnz1, feat, hidden);
    for (int i = 0; i < nhid; i++) if (hidden[i] < 0.f) hidden[i] = 0.f;

    /* Layer 2: sparse matmul → [nout], no ReLU (linear output) */
    mhr_spmv_relu(d->corr_sp2_row, d->corr_sp2_col, d->corr_sp2_val, nnz2, hidden, out);

    /* Add to unposed vertices */
    for (int i = 0; i < nout; i++) unposed[i] += out[i];

    free(feat); free(hidden); free(out);
    return 1;
}

/* ── main per-frame compute ───────────────────────────────────────────────── */

/* mhr_lbs_compute — full LBS forward pass, replicating body_model.pt:
 *
 *  model_params [204] layout (see build_model_params in preprocess.hpp):
 *    [0:3]    global_trans × 10  (zeroed for single-view)
 *    [3:6]    global_rot   Euler ZYX (rx, ry, rz) extracted from the 6D head output
 *    [6:136]  body_pose_params [130 of 133]  Euler ZYX per DOF (hands zeroed)
 *    [136:204] scales
 *
 *  Pipeline stages:
 *    1. unposed = base_shape + Σ shape_coeff[i]*shape_vec[i] + Σ face_coeff[i]*face_vec[i]
 *    2. joint_params [n_joints×7] = PT [n_joints×7, pt_cols] @ model_params[:pt_cols]
 *       Each joint row: [tx, ty, tz, rx, ry, rz, log2_scale]
 *       rx,ry,rz are in PyMomentum XYZ intrinsic convention: R = Rz(rz)*Ry(ry)*Rx(rx).
 *    3. local TRS per joint:
 *         t_local[j] = joint_offsets[j] + jp[j][0:3]
 *         q_local[j] = joint_prerotations[j] * euler_to_quat(jp[j][3:6])
 *                      (pre-rotation establishes rest-frame; euler applied on top)
 *         s_local[j] = exp2(jp[j][6])
 *    4. FK (parent index < child index guaranteed by sorted order):
 *         g_q[j] = g_q[parent] * q_local[j]   ← global = parent_global ∘ local
 *         g_t[j] = g_t[parent] + g_s[parent] * rotate(g_q[parent], t_local[j])
 *         g_s[j] = g_s[parent] * s_local[j]
 *    5. skin TRS = global(j) ∘ inv_bind(j):
 *         skin_q[j] = g_q[j] * ib_q[j]
 *         skin_t[j] = g_t[j] + g_s[j] * rotate(g_q[j], ib_t[j])
 *         skin_s[j] = g_s[j] * ib_s[j]
 *    6. LBS: out_vert[vi] += Σ_j w[j,vi] * (skin_t[j] + skin_s[j]*rotate(skin_q[j], unposed[vi]))
 *
 *  On first call, model_params and key intermediate values are written to
 *  /tmp/mhr_lbs_dump.bin for comparison with the Python dumper:
 *    dump_joint_transforms.py --lbs body_model.lbs --dump /tmp/mhr_lbs_dump.bin
 */
int mhr_lbs_compute(const struct MHR_LBS_Data *d,
                    const float *model_params,  /* [204]        */
                    const float *shape_coeffs,  /* [n_shape_pc] */
                    const float *face_coeffs,   /* [n_face_pc]  */
                    float       *out_verts,     /* [n_verts*3], caller-alloc */
                    float       *out_joints)    /* [n_joints*3], optional */
{
    if (!d || !model_params || !shape_coeffs || !face_coeffs || !out_verts)
        return 0;

    int nj  = d->n_joints;
    int ns  = d->n_skin;
    int nv  = d->n_verts;
    int npr = d->pt_rows;
    int npc = d->pt_cols;

    /* First-frame dump: write model_params [204 floats] to /tmp/mhr_lbs_dump.bin
     * so dump_joint_transforms.py can reproduce and verify every stage offline.
     * Opt-in only — set FSB_LBS_DUMP=1 to enable (off by default). */
    { static int dumped = 0;
      if (!dumped && getenv("FSB_LBS_DUMP")) {
          dumped = 1;
          FILE *fp = fopen("/tmp/mhr_lbs_dump.bin","wb");
          if (fp) {
              /* header: n_joints, pt_cols so Python can replicate Step 2 */
              int hdr[2] = {nj, npc};
              fwrite(hdr, sizeof(int), 2, fp);
              fwrite(model_params,  sizeof(float), 204,               fp);
              fwrite(shape_coeffs,  sizeof(float), (size_t)d->n_shape_pc, fp);
              fwrite(face_coeffs,   sizeof(float), (size_t)d->n_face_pc,  fp);
              fclose(fp);
              fprintf(stderr,"[LBS] wrote first-frame dump to /tmp/mhr_lbs_dump.bin\n");
          }
      }
    }

    /* Step 1 — unposed vertices: base_shape + shape_bs + face_bs */
    float *unposed = (float*)malloc((size_t)nv * 3 * sizeof(float));
    if (!unposed) return 0;
    memcpy(unposed, d->base_shape, (size_t)nv * 3 * sizeof(float));

#if 0 /* DEBUG: base_shape bounds — re-enable to diagnose mesh scale issues */
    // Debug: base_shape bounds
    { float bx=1e9f,by=1e9f,bz=1e9f;
      for(int i=0;i<nv*3;i+=3) { if(d->base_shape[i]<bx)bx=d->base_shape[i]; if(d->base_shape[i+1]<by)by=d->base_shape[i+1]; if(d->base_shape[i+2]<bz)bz=d->base_shape[i+2]; }
      float tx=-1e9f,ty=-1e9f,tz=-1e9f;
      for(int i=0;i<nv*3;i+=3) { if(d->base_shape[i]>tx)tx=d->base_shape[i]; if(d->base_shape[i+1]>ty)ty=d->base_shape[i+1]; if(d->base_shape[i+2]>tz)tz=d->base_shape[i+2]; }
      fprintf(stderr,"[LBS] base_shape: x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]\n",bx,tx,by,ty,bz,tz);
    }
#endif

    for (int i = 0; i < d->n_shape_pc; i++) {
        float c = shape_coeffs[i];
        if (c == 0.f) continue;
        const float *sv = d->shape_vectors + (size_t)i * nv * 3;
        for (int k = 0; k < nv*3; k++) unposed[k] += c * sv[k];
    }
    for (int i = 0; i < d->n_face_pc; i++) {
        float c = face_coeffs[i];
        if (c == 0.f) continue;
        const float *fv = d->face_vectors + (size_t)i * nv * 3;
        for (int k = 0; k < nv*3; k++) unposed[k] += c * fv[k];
    }

#if 0 /* DEBUG: unposed bounds — re-enable to verify shape blend */
    // Debug: unposed bounds
    { float ux=1e9f,uy=1e9f,uz=1e9f;
      for(int i=0;i<nv*3;i+=3) { if(unposed[i]<ux)ux=unposed[i]; if(unposed[i+1]<uy)uy=unposed[i+1]; if(unposed[i+2]<uz)uz=unposed[i+2]; }
      float vx=-1e9f,vy=-1e9f,vz=-1e9f;
      for(int i=0;i<nv*3;i+=3) { if(unposed[i]>vx)vx=unposed[i]; if(unposed[i+1]>vy)vy=unposed[i+1]; if(unposed[i+2]>vz)vz=unposed[i+2]; }
      fprintf(stderr,"[LBS] unposed:    x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]\n",ux,vx,uy,vy,uz,vz);
    }
#endif

#if 0 /* DEBUG: incoming coefficient magnitudes — re-enable to check input range */
    /* Debug: incoming coefficient magnitudes */
    { float mp_max=0, sc_max=0, fc_max=0;
      for(int i=0;i<204;i++) { float a=model_params[i]; if(a<0)a=-a; if(a>mp_max)mp_max=a; }
      for(int i=0;i<d->n_shape_pc;i++) { float a=shape_coeffs[i]; if(a<0)a=-a; if(a>sc_max)sc_max=a; }
      for(int i=0;i<d->n_face_pc;i++) { float a=face_coeffs[i]; if(a<0)a=-a; if(a>fc_max)fc_max=a; }
      fprintf(stderr,"[LBS] input max abs: model_params=%.4f shape=%.4f face=%.4f  npc=%d\n", mp_max, sc_max, fc_max, npc);
    }
#endif

    /* Step 2 — joint_params = PT [npr×npc] @ concat(model_params[204], zeros) */
    float *joint_params = (float*)malloc((size_t)npr * sizeof(float));
    float *input_vec    = (float*)calloc((size_t)npc,  sizeof(float));
    if (!joint_params || !input_vec) {
        free(joint_params); free(input_vec); free(unposed); return 0;
    }
    memcpy(input_vec, model_params, (size_t)(npc < 204 ? npc : 204) * sizeof(float));

    for (int j = 0; j < npr; j++) {
        float acc = 0.f;
        const float *row = d->PT + (size_t)j * npc;
        for (int k = 0; k < npc; k++) acc += row[k] * input_vec[k];
        joint_params[j] = acc;
    }
    free(input_vec);

    /* Step 2b — pose correctives: add corrective offsets to unposed (before LBS).
     * Requires joint_params (computed above) which is freed in Step 3. */
    mhr_apply_correctives(d, joint_params, unposed);

    /* Step 3 — local skeleton state per joint: (t_local, q_local, s_local)
     *
     * joint_params row j: [tx, ty, tz, rx, ry, rz, log2_scale]
     *   tx,ty,tz  are ADDITIVE offsets relative to joint_offsets[j] (rest-pose position).
     *   rx,ry,rz  are ZYX Euler angles (PyMomentum XYZ intrinsic: R = Rz(rz)*Ry(ry)*Rx(rx)).
     *   log2_scale is decoded as exp2(val) = 2^val.
     *
     * q_local[j] = joint_prerotations[j] * euler_to_quat(rx,ry,rz)
     *   joint_prerotations establishes each joint's rest-pose orientation.
     *   The pose quaternion (from Euler) is applied ON TOP via right-multiplication.
     *   R_local = R_pre @ R_euler  (pre-rotation first, then the driven pose).
     */
    float *t_local  = (float*)malloc((size_t)nj * 3 * sizeof(float));
    float *q_local  = (float*)malloc((size_t)nj * 4 * sizeof(float));
    float *s_local  = (float*)malloc((size_t)nj * sizeof(float));
    if (!t_local || !q_local || !s_local) { free(t_local); free(q_local); free(s_local); return 0; }

    for (int j = 0; j < nj; j++) {
        const float *jp  = joint_params + j * 7;
        const float *off = d->joint_offsets      + j * 3;
        const float *pre = d->joint_prerotations + j * 4;  /* XYZW */

        /* rest-pose offset + driven translation */
        t_local[j*3+0] = off[0] + jp[0];
        t_local[j*3+1] = off[1] + jp[1];
        t_local[j*3+2] = off[2] + jp[2];

        /* jp[3]=rx jp[4]=ry jp[5]=rz  →  q representing Rz(rz)*Ry(ry)*Rx(rx)
         * Then compose: q_local = pre * q_euler  so R_local = R_pre @ R_euler. */
        float q_euler[4];
        mhr_euler_xyz_to_quat(jp[3], jp[4], jp[5], q_euler);
        mhr_qmul(q_local + j*4, pre, q_euler);  /* R_local = R_pre @ R_euler */

        s_local[j] = expf(jp[6] * MHR_LN2);  /* exp2(log2_scale) */
    }
    free(joint_params);

    /* Step 4 — FK chain: accumulate global TRS from root to leaves.
     *
     * joint_parents is sorted so that parent index < child index, meaning a
     * single forward pass is sufficient (no dependency ordering needed).
     *
     * For joint j with parent p:
     *   g_q[j] = g_q[p] * q_local[j]     → R_global[j] = R_global[p] @ R_local[j]
     *   g_t[j] = g_t[p] + g_s[p] * R_global[p] @ t_local[j]   (offset rotated by parent)
     *   g_s[j] = g_s[p] * s_local[j]
     *
     * For the root joint (parent < 0), global = local directly.
     */
   float *g_t = (float*)malloc((size_t)nj * 3 * sizeof(float));
    float *g_q = (float*)malloc((size_t)nj * 4 * sizeof(float));
    float *g_s = (float*)malloc((size_t)nj * sizeof(float));
    if (!g_t || !g_q || !g_s) { free(g_t); free(g_q); free(g_s); free(t_local); free(q_local); free(s_local); return 0; }

    //fprintf(stderr,"[LBS] root joint: t=(%.4f,%.4f,%.4f) s=%.6f\n",
    //        t_local[0], t_local[1], t_local[2], s_local[0]);

    /* Check for NaN/Inf in local transforms */
    { int bad = 0;
      for(int j=0;j<nj;j++) {
          for(int c=0;c<4;c++) { if(!isfinite(q_local[j*4+c])) bad++; }
          if(!isfinite(s_local[j])) bad++;
      }
      if(bad) fprintf(stderr,"[LBS] WARNING: %d NaN/Inf in local transforms\n", bad);
    }

    for (int j = 0; j < nj; j++) {
        int p = d->joint_parents[j];
        if (p < 0) {
            /* root: global == local */
            memcpy(g_t + j*3, t_local + j*3, 3*sizeof(float));
            memcpy(g_q + j*4, q_local + j*4, 4*sizeof(float));
            g_s[j] = s_local[j];
        } else {
            g_s[j] = g_s[p] * s_local[j];
            /* g_q[j] = g_q[p] * q_local[j]  →  R_global[j] = R_parent @ R_local[j] */
            mhr_qmul(g_q + j*4, g_q + p*4, q_local + j*4);
            /* child offset rotated by parent global rotation, then scaled and added */
            float rt[3];
            mhr_qrot(rt, g_q + p*4, t_local + j*3);
            g_t[j*3+0] = g_t[p*3+0] + g_s[p] * rt[0];
            g_t[j*3+1] = g_t[p*3+1] + g_s[p] * rt[1];
            g_t[j*3+2] = g_t[p*3+2] + g_s[p] * rt[2];
        }
    }

    //fprintf(stderr,"[LBS] FK loop done\n");

    /* Step 5 — skin TRS = global(j) ∘ inv_bind(j)
     *
     * inv_bind_pose[j] is the inverse of the joint's bind-pose transform.
     * Composing global(j) with inv_bind(j) gives the net deformation applied
     * to a vertex skinned to joint j when it moves from rest to the current pose.
     *
     * inv_bind layout per joint (8 floats): [tx, ty, tz, qx, qy, qz, qw, scale]
     *
     * Composition (TRS concatenation):
     *   skin_q = g_q @ ib_q    → R_skin = R_global @ R_inv_bind
     *   skin_t = g_t + g_s * rotate(g_q, ib_t)
     *   skin_s = g_s * ib_scale
     */
    float *skin_t = (float*)malloc((size_t)nj * 3 * sizeof(float));
    float *skin_q = (float*)malloc((size_t)nj * 4 * sizeof(float));
    float *skin_s = (float*)malloc((size_t)nj * sizeof(float));
    if (!skin_t || !skin_q || !skin_s) { free(skin_t); free(skin_q); free(skin_s); free(g_t); free(g_q); free(g_s); free(t_local); free(q_local); free(s_local); return 0; }

    for (int j = 0; j < nj; j++) {
        const float *ib  = d->inv_bind_pose + j * 8;  /* [tx,ty,tz, qx,qy,qz,qw, scale] */
        float        ibs = ib[7];

        skin_s[j] = g_s[j] * ibs;
        mhr_qmul(skin_q + j*4, g_q + j*4, ib + 3);  /* R_skin = R_global @ R_inv_bind */
        float rt[3];
        mhr_qrot(rt, g_q + j*4, ib);   /* rotate inv_bind translation by global orientation */
        skin_t[j*3+0] = g_t[j*3+0] + g_s[j] * rt[0];
        skin_t[j*3+1] = g_t[j*3+1] + g_s[j] * rt[1];
        skin_t[j*3+2] = g_t[j*3+2] + g_s[j] * rt[2];
    }

    //fprintf(stderr,"[LBS] Skin TRS done\n");
    { int bad2 = 0;
      for(int j=0;j<nj;j++) {
          for(int c=0;c<4;c++) { if(!isfinite(skin_q[j*4+c])) bad2++; }
          if(!isfinite(skin_s[j])) bad2++;
      }
      if(bad2) fprintf(stderr,"[LBS] WARNING: %d NaN/Inf in skin transforms\n", bad2);
    }

    /* Step 6 — LBS: sparse weighted accumulation */
    memset(out_verts, 0, (size_t)nv * 3 * sizeof(float));

    //fprintf(stderr,"[LBS] LBS loop start, ns=%d\n", ns);
#if 0 /* DEBUG: skin index range — re-enable to verify skin data integrity */
    { int ji_max=0, vi_max=0;
      for(int k=0;k<ns;k++) { if(d->skin_joint_idx[k]>ji_max) ji_max=d->skin_joint_idx[k]; if(d->skin_vert_idx[k]>vi_max) vi_max=d->skin_vert_idx[k]; }
      fprintf(stderr,"[LBS] skin indices: ji_max=%d/127 vi_max=%d/%d\n", ji_max, vi_max, nv);
    }
#endif

    for (int k = 0; k < ns; k++) {
        int   ji = d->skin_joint_idx[k];
        int   vi = d->skin_vert_idx[k];
        float w  = d->skin_weights[k];
        float sx = skin_s[ji];

        const float *vr = unposed + vi * 3;
        float pv[3];
        mhr_qrot(pv, skin_q + ji*4, vr);
        out_verts[vi*3+0] += w * (skin_t[ji*3+0] + sx * pv[0]);
        out_verts[vi*3+1] += w * (skin_t[ji*3+1] + sx * pv[1]);
        out_verts[vi*3+2] += w * (skin_t[ji*3+2] + sx * pv[2]);
    }

    //fprintf(stderr,"[LBS] LBS loop done\n");
    //fprintf(stderr,"[LBS] freeing unposed\n");
    free(unposed);

    /* Step 7 — coordinate flip + cm→m.
     * Body model data is stored in cm; Python mhr_forward divides by 100 to get metres.
     * Apply the same conversion here so C output matches Python mhr_forward output. */
    for (int v = 0; v < nv; v++) {
        out_verts[v*3+0] *=  0.01f;
        out_verts[v*3+1] *= -0.01f;   /* Y flip + cm→m */
        out_verts[v*3+2] *= -0.01f;   /* Z flip + cm→m */
    }

    //fprintf(stderr,"[LBS] coord flip done\n");

    /* Step 8 — output joint world positions with Y,Z flip + cm→m. */
    if (out_joints) {
        for (int j = 0; j < nj; j++) {
            out_joints[j*3+0] =  g_t[j*3+0] * 0.01f;
            out_joints[j*3+1] = -g_t[j*3+1] * 0.01f;
            out_joints[j*3+2] = -g_t[j*3+2] * 0.01f;
        }
    }

    //fprintf(stderr,"[LBS] output joints done, returning\n");

    free(t_local); free(q_local); free(s_local);
    free(g_t); free(g_q); free(g_s);
    free(skin_t); free(skin_q); free(skin_s);
    return 1;
}
