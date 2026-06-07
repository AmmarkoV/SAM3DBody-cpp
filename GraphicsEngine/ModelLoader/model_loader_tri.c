/** @file model_loader_tri.c
 *  @brief  TRIModels loader/writer and basic functions
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include "model_loader_tri.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//For lowercase
#include <ctype.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#if INCLUDE_OPENGL_CODE
 #include <GL/gl.h>
 #include <GL/glx.h>    /* this includes the necessary X headers */
#endif // INCLUDE_OPENGL_CODE

int tri_warnIncompleteReads(const char * msg,unsigned int expected,unsigned int recvd)
{
   if (expected!=recvd) { fprintf(stderr,YELLOW "Incomplete read of %s ( %u/%u ) \n" NORMAL , msg,recvd,expected); return 1; }
   return 0;
}

void print4x4FMatrixTRI(const char * str , float * matrix4x4)
{
  fprintf( stderr, "  %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "  %f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f ",matrix4x4[2]);  fprintf( stderr, "%f\n",matrix4x4[3]);
  fprintf( stderr, "  %f ",matrix4x4[4]);  fprintf( stderr, "%f ",matrix4x4[5]);  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f\n",matrix4x4[7]);
  fprintf( stderr, "  %f ",matrix4x4[8]);  fprintf( stderr, "%f ",matrix4x4[9]);  fprintf( stderr, "%f ",matrix4x4[10]); fprintf( stderr, "%f\n",matrix4x4[11]);
  fprintf( stderr, "  %f ",matrix4x4[12]); fprintf( stderr, "%f ",matrix4x4[13]); fprintf( stderr, "%f ",matrix4x4[14]); fprintf( stderr, "%f\n",matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");

}

void printTRIModel(struct TRI_Model * triModel)
{
 fprintf(stderr,"Number Of Vertices  %u \n",triModel->header.numberOfVertices);
 fprintf(stderr,"Number Of Normals  %u \n",triModel->header.numberOfNormals);
 fprintf(stderr,"Number Of TextureCoords  %u \n",triModel->header.numberOfTextureCoords);
 fprintf(stderr,"Number Of Colors  %u \n",triModel->header.numberOfColors);
 fprintf(stderr,"Number Of Indices  %u \n",triModel->header.numberOfIndices);
 fprintf(stderr,"Number Of Bones  %u \n",triModel->header.numberOfBones);
}

void printTRIBoneStructure(struct TRI_Model * triModel, int alsoPrintMatrices)
{
  unsigned int k=0, i=0 , parent , child ;
   for (i=0; i<triModel->header.numberOfBones; i++)
   {
     fprintf(stderr,"Bone %u : ",i);
     fprintf(stderr,GREEN "`%s` \n" NORMAL,triModel->bones[i].boneName);
     if (triModel->bones[i].info!=0)
     {
      fprintf(stderr," Weights Number %u \n",triModel->bones[i].info->boneWeightsNumber);
      parent = triModel->bones[i].info->boneParent;
      fprintf(stderr," Parent : %u %s \n",parent,triModel->bones[parent].boneName);

      if (triModel->bones[i].boneChild!=0)
      {
      fprintf(stderr," Children (%u) : " , triModel->bones[i].info->numberOfBoneChildren);
       for (k=0; k<triModel->bones[i].info->numberOfBoneChildren; k++)
        {
         child=triModel->bones[i].boneChild[k];
         fprintf(stderr,"%u %s , ",child,triModel->bones[child].boneName);
        }
       fprintf(stderr,"\n");
      }

     if (alsoPrintMatrices)
        {
         print4x4FMatrixTRI("matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose", triModel->bones[i].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose );
         print4x4FMatrixTRI("finalVertexTransformation", triModel->bones[i].info->finalVertexTransformation );
         print4x4FMatrixTRI("localTransformation", triModel->bones[i].info->localTransformation );
        }
     }
   }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

struct TRI_Model * tri_allocateModel()
{
  struct TRI_Model * newModel = (struct TRI_Model * ) malloc(sizeof(struct TRI_Model));
  if (newModel!=0) // Clear new model if it was allocated..
        { memset(newModel,0,sizeof(struct TRI_Model)); }
  return (struct TRI_Model * ) newModel;
}

struct TRI_Model * allocateModelTri()
{
    fprintf(stderr,YELLOW " allocateModelTri is deprecated \n" NORMAL);
    return tri_allocateModel();
}

void tri_deallocModelInternals(struct TRI_Model * triModel)
{
  if (triModel==0) { return ; }

  triModel->header.numberOfVertices = 0;
  if (triModel->vertices!=0) { free(triModel->vertices); triModel->vertices=0; }

  triModel->header.nameSize = 0;
  if (triModel->name!=0) { free(triModel->name); triModel->name=0; }

  triModel->header.numberOfNormals = 0;
  if (triModel->normal!=0) { free(triModel->normal); triModel->normal=0; }

  triModel->header.numberOfColors = 0;
  if (triModel->colors!=0) { free(triModel->colors); triModel->colors=0; }

  triModel->header.numberOfTextureCoords = 0;
  if (triModel->textureCoords!=0) { free(triModel->textureCoords); triModel->textureCoords=0; }

  triModel->header.numberOfIndices = 0;
  if (triModel->indices!=0) { free(triModel->indices); triModel->indices=0; }

   if  (
           (triModel->header.numberOfBones>0) &&
           (triModel->bones!=0)
       )
      {
        unsigned int boneNum =0;
        for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
        {
          if (triModel->bones[boneNum].boneChild!=0)
           { free(triModel->bones[boneNum].boneChild); triModel->bones[boneNum].boneChild=0; }

          if (triModel->bones[boneNum].info!=0)
           { free(triModel->bones[boneNum].info); triModel->bones[boneNum].info=0; }

          if (triModel->bones[boneNum].boneName!=0)
           { free(triModel->bones[boneNum].boneName); triModel->bones[boneNum].boneName=0; }

          if (triModel->bones[boneNum].weightValue!=0)
           { free(triModel->bones[boneNum].weightValue); triModel->bones[boneNum].weightValue=0; }

          if (triModel->bones[boneNum].weightIndex!=0)
           { free(triModel->bones[boneNum].weightIndex); triModel->bones[boneNum].weightIndex=0; }
        }
      }

   if (triModel->bones!=0)         { free(triModel->bones); triModel->bones=0; }



  triModel->header.textureDataWidth = 0;
  triModel->header.textureDataHeight = 0;
  triModel->header.textureDataChannels = 0;
  if ( (triModel->textureData!=0) && (triModel->header.textureUploadedToGPU) )
  {
    fprintf(stderr,"texture contains GPU pointer, not losing it..\n");
    //triModel->header.textureBindGLBuffer;
    //triModel->header.textureUploadedToGPU;
  }
  if (triModel->textureData!=0) { free(triModel->textureData); triModel->textureData=0; }

   //Make sure everything is wiped clean..
   memset(triModel,0,sizeof(struct TRI_Model));
}


void deallocInternalsOfModelTri(struct TRI_Model * triModel)
{
    fprintf(stderr,YELLOW " deallocInternalsOfModelTri is deprecated \n" NORMAL);
    //Deprecated call..
    return tri_deallocModelInternals(triModel);
}

int tri_freeModel(struct TRI_Model * triModel)
{
  if (triModel!=0)
  {
   tri_deallocModelInternals(triModel);
   free(triModel);
  }
 return 1;
}

int freeModelTri(struct TRI_Model * triModel)
{
    fprintf(stderr,YELLOW " freeModelTri is deprecated \n" NORMAL);
  //Deprecated call..
  return tri_freeModel(triModel);
}


//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------


void tri_wipeModel(struct TRI_Model * triModel)
{
  if (triModel==0)  { return; }
  //TODO deallocate stuff
  memset(triModel,0,sizeof(struct TRI_Model));
}



void tri_copyModelHeader(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN )
{
  if (triModelOUT==0) { return; }
  if (triModelIN==0)  { return; }
  //fprintf(stderr,"Cleaning output model..\n");
  //memset(triModelOUT,0,sizeof(struct TRI_Model));
  //fprintf(stderr,"Copying header..\n");
  memcpy(&triModelOUT->header , &triModelIN->header , sizeof(struct TRI_Header));

 return;
}



int tri_doBoneDeepCopy(
                       struct TRI_Model * triModelOUT,
                       struct TRI_Model * triModelIN,
                       TRIBoneID boneOut,
                       TRIBoneID boneIn
                      )
{
  if (triModelOUT==0) { return 0; }
  if (triModelIN==0)  { return 0; }
  if (boneOut<triModelOUT->header.numberOfBones) { return 0; }
  if (boneIn<triModelIN->header.numberOfBones)   { return 0; }

  //First read dimensions of bone string and the number of weights for the bone..
  if (triModelOUT->bones[boneOut].info!=0) { free(triModelOUT->bones[boneOut].info); }
  triModelOUT->bones[boneOut].info = (struct TRI_Bones_Header*) malloc(sizeof(struct TRI_Bones_Header));
  memcpy( triModelOUT->bones[boneOut].info , triModelIN->bones[boneIn].info , sizeof(struct TRI_Bones_Header) );

  //Allocate enough space for the bone string , read it  , and null terminate it
  unsigned int itemSize = sizeof(char);
  unsigned int count = triModelIN->bones[boneIn].info->boneNameSize;
  if (triModelOUT->bones[boneOut].boneName!=0) { free(triModelOUT->bones[boneOut].boneName); }
  triModelOUT->bones[boneOut].boneName = ( char * ) malloc ( (itemSize)*(count+1) );
  memcpy( triModelOUT->bones[boneOut].boneName , triModelIN->bones[boneIn].boneName , itemSize*count );
  triModelOUT->bones[boneOut].boneName[triModelIN->bones[boneIn].info->boneNameSize]=0; //Null terminator..

  //Allocate enough space for the weight values , and read them
  itemSize = sizeof(float);        count = triModelIN->bones[boneIn].info->boneWeightsNumber;
  if (triModelOUT->bones[boneOut].weightValue!=0) { free(triModelOUT->bones[boneOut].weightValue); }
  triModelOUT->bones[boneOut].weightValue = ( float * ) malloc ( itemSize * count );
  memcpy( triModelOUT->bones[boneOut].weightValue , triModelIN->bones[boneIn].weightValue , itemSize * count );

  //Allocate enough space for the weight indexes , and read them
  itemSize = sizeof(unsigned int); count = triModelIN->bones[boneIn].info->boneWeightsNumber;
  if (triModelOUT->bones[boneOut].weightIndex!=0) { free(triModelOUT->bones[boneOut].weightIndex); }
  triModelOUT->bones[boneOut].weightIndex = ( unsigned int * ) malloc ( itemSize * count );
  memcpy( triModelOUT->bones[boneOut].weightIndex , triModelIN->bones[boneIn].weightIndex , itemSize * count );

  itemSize = sizeof(unsigned int); count = triModelIN->bones[boneIn].info->numberOfBoneChildren;
  if (triModelOUT->bones[boneOut].boneChild!=0)
            {
              free(triModelOUT->bones[boneOut].boneChild);
              triModelOUT->bones[boneOut].boneChild=0;
            }

  if ( ( triModelIN->bones[boneIn].boneChild !=0 ) && (count>0) )
          {
            triModelOUT->bones[boneOut].boneChild = (unsigned int *) malloc(  itemSize * (count+1) );
            if (triModelOUT->bones[boneOut].boneChild !=0 )
             {
              memcpy( triModelOUT->bones[boneOut].boneChild , triModelIN->bones[boneIn].boneChild ,  itemSize * count );
             } else
             {
              fprintf(stderr,RED "Cannot allocate enough memory for bone %u children..\n" NORMAL,boneOut);
             }
          }

  return 1;
}


void tri_copyBones(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN)
{
  if ( (triModelIN->bones!=0) && (triModelIN->header.numberOfBones>0) )
  {
     triModelOUT->header.numberOfBones = triModelIN->header.numberOfBones;
     triModelOUT->bones = (struct TRI_Bones *) malloc(sizeof(struct TRI_Bones) * triModelIN->header.numberOfBones);
     memset(triModelOUT->bones, 0 , sizeof(struct TRI_Bones) * triModelIN->header.numberOfBones);

     TRIBoneID boneNum=0;
     for (boneNum=0; boneNum<triModelIN->header.numberOfBones; boneNum++)
        {
          tri_doBoneDeepCopy(
                             triModelOUT,
                             triModelIN,
                             boneNum,
                             boneNum
                           );
        }
  }
}


int tri_deepCopyBoneValuesButNotStructure(struct TRI_Model * target,struct TRI_Model  * source)
{
 if ( (source==0) || (target==0) ) { return 0; }

 unsigned int unresolvedBones = 0;
 for (TRIBoneID targetBoneID=0; targetBoneID<source->header.numberOfBones; targetBoneID++)
 {
   TRIBoneID sourceBoneID=targetBoneID;
   //fprintf(stderr,"Want data for %s(%u) : ",source->bones[targetBoneID].boneName,targetBoneID);
   if ( tri_findBone(source,source->bones[targetBoneID].boneName,&sourceBoneID) )
          {
            //fprintf(stderr,"Will copy from source (%u) \n",sourceBoneID);
            tri_doBoneDeepCopy(target,source,targetBoneID,sourceBoneID);
          } else
          {
              fprintf(stderr,RED "Could not resolve %s(%u) \n"NORMAL,source->bones[targetBoneID].boneName,targetBoneID);
              ++unresolvedBones;
          }
 }

 return (unresolvedBones==0);
}

int tri_flattenIndexedModel(struct TRI_Model * triModel,struct TRI_Model * indexed)
{
    //fprintf(stderr,YELLOW "fillFlatModelTriFromIndexedModelTri(%p,%p)..!\n" NORMAL,triModel,indexed);
    if ( (triModel==0) || (indexed==0) )
    {
      fprintf(stderr,RED "\nerror : Cannot flatten with null models..\n" NORMAL);
      return 0;
    }

    if ( (indexed->indices==0) || (indexed->header.numberOfIndices==0) )
    {
      fprintf(stderr,RED "\nerror : Supposedly indexed model is not indexed..!\n" NORMAL);
      tri_copyModel(triModel,indexed,1,0);
      return 1;
    }


    if (triModel==indexed)
    {
      //User requested to do flattening providing the same source and target..!
      //Do our own temporary stack allocation and go on with this call..
      struct TRI_Model * temporary = allocateModelTri();
      if (temporary!=0)
      {
       tri_copyModel(temporary,indexed,1,0);
       int result = fillFlatModelTriFromIndexedModelTri(triModel,temporary);
       tri_freeModel(temporary);
       temporary=0;
       return result;
      }
      return 0;
    }

    //Get rid of old data..
    //===================================================================
    tri_deallocModelInternals(triModel);

    //Write header..
    //===================================================================
    triModel->header.triType = TRI_LOADER_VERSION;
    triModel->header.floatSize = sizeof(float);
    triModel->header.nameSize = indexed->header.nameSize;
    triModel->header.TRIMagic[0] = 'T';
    triModel->header.TRIMagic[1] = 'R';
    triModel->header.TRIMagic[2] = 'I';
    triModel->header.TRIMagic[3] = '3';
    triModel->header.TRIMagic[4] = 'D';
    triModel->header.numberOfVertices      = indexed->header.numberOfIndices*3;
    triModel->header.numberOfNormals       = indexed->header.numberOfIndices*3;
    triModel->header.numberOfTextureCoords = indexed->header.numberOfIndices*2;
    triModel->header.numberOfColors        = indexed->header.numberOfIndices*3;
    triModel->header.numberOfIndices       = 0;
    triModel->header.numberOfBones         = 0; //This will get filled in later
    //We don't care about anything texture related but we must propagate the GL buffer TextureID
    triModel->header.textureBindGLBuffer   = indexed->header.textureBindGLBuffer;
    triModel->header.textureUploadedToGPU  = indexed->header.textureUploadedToGPU;

    //fprintf(stderr,YELLOW "\nwarning : Flattening a model loses its bone structure for now..\n" NORMAL);


    triModel->name = (char * ) malloc( (triModel->header.nameSize+1)  * sizeof(char));
    if (triModel->name!=0)
    {
      memcpy(triModel->name , indexed->name , triModel->header.nameSize);
      triModel->name[triModel->header.nameSize]=0;
    }

    //Allocate space for new data..
    //===================================================================
	triModel->vertices       = (float*) malloc( triModel->header.numberOfVertices      *3 *3     * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals       *3 *3     * sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTextureCoords *3 *2     * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors        *3 *3     * sizeof(float));
    triModel->indices        = 0; //Flat tri models dont have indices
    triModel->bones          = 0; //Bones will get filled in later

    unsigned int o=0,n=0,t=0,c=0;
    //---------------------------
	for (unsigned int indexID=0; indexID<indexed->header.numberOfIndices/3; indexID++)
    {
		unsigned int faceTriA_ID = indexed->indices[(indexID*3)+0];
		unsigned int faceTriB_ID = indexed->indices[(indexID*3)+1];
		unsigned int faceTriC_ID = indexed->indices[(indexID*3)+2];
        //------------------------------------------------
        unsigned int faceTriA_X = (faceTriA_ID*3)+0;
        unsigned int faceTriA_Y = (faceTriA_ID*3)+1;
        unsigned int faceTriA_Z = (faceTriA_ID*3)+2;
        //---------------------------------------
        unsigned int faceTriB_X = (faceTriB_ID*3)+0;
        unsigned int faceTriB_Y = (faceTriB_ID*3)+1;
        unsigned int faceTriB_Z = (faceTriB_ID*3)+2;
        //---------------------------------------
        unsigned int faceTriC_X = (faceTriC_ID*3)+0;
        unsigned int faceTriC_Y = (faceTriC_ID*3)+1;
        unsigned int faceTriC_Z = (faceTriC_ID*3)+2;

		//fprintf(stderr,"%u / %u \n" , o , triModel->header.numberOfVertices * 3 );
	    triModel->vertices[o++] = indexed->vertices[faceTriA_X];
	    triModel->vertices[o++] = indexed->vertices[faceTriA_Y];
	    triModel->vertices[o++] = indexed->vertices[faceTriA_Z];
        //-----------------------------------------------------
	    triModel->vertices[o++] = indexed->vertices[faceTriB_X];
	    triModel->vertices[o++] = indexed->vertices[faceTriB_Y];
	    triModel->vertices[o++] = indexed->vertices[faceTriB_Z];
        //-----------------------------------------------------
	    triModel->vertices[o++] = indexed->vertices[faceTriC_X];
	    triModel->vertices[o++] = indexed->vertices[faceTriC_Y];
	    triModel->vertices[o++] = indexed->vertices[faceTriC_Z];

      if (indexed->normal)
        {
			triModel->normal[n++] = indexed->normal[faceTriA_X];
			triModel->normal[n++] = indexed->normal[faceTriA_Y];
			triModel->normal[n++] = indexed->normal[faceTriA_Z];
            //-------------------------------------------------
			triModel->normal[n++] = indexed->normal[faceTriB_X];
			triModel->normal[n++] = indexed->normal[faceTriB_Y];
			triModel->normal[n++] = indexed->normal[faceTriB_Z];
            //-------------------------------------------------
			triModel->normal[n++] = indexed->normal[faceTriC_X];
			triModel->normal[n++] = indexed->normal[faceTriC_Y];
			triModel->normal[n++] = indexed->normal[faceTriC_Z];
		}

      if (indexed->textureCoords)
        {
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriA_ID*2)+0];
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriA_ID*2)+1];
            //-------------------------------------------------------------------
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriB_ID*2)+0];
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriB_ID*2)+1];
            //-------------------------------------------------------------------
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriC_ID*2)+0];
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriC_ID*2)+1];
		}

      if (indexed->colors)
         {
          triModel->colors[c++] = indexed->colors[faceTriA_X];
          triModel->colors[c++] = indexed->colors[faceTriA_Y];
          triModel->colors[c++] = indexed->colors[faceTriA_Z];
            //-----------------------------------------------
          triModel->colors[c++] = indexed->colors[faceTriB_X];
          triModel->colors[c++] = indexed->colors[faceTriB_Y];
          triModel->colors[c++] = indexed->colors[faceTriB_Z];
            //-----------------------------------------------
          triModel->colors[c++] = indexed->colors[faceTriC_X];
          triModel->colors[c++] = indexed->colors[faceTriC_Y];
          triModel->colors[c++] = indexed->colors[faceTriC_Z];
        }
	}

  //Also copy bone structures..
  tri_copyBones(triModel,indexed);

 return 1;
}

int tri_flattenIndexedModelInPlace(struct TRI_Model * indexedInputThatWillBecomeFlatOutput)
{
   return tri_flattenIndexedModel(indexedInputThatWillBecomeFlatOutput,indexedInputThatWillBecomeFlatOutput);
}

int fillFlatModelTriFromIndexedModelTri(struct TRI_Model * flatOutput,struct TRI_Model * indexedInput)
{
   return tri_flattenIndexedModel(flatOutput,indexedInput);
}



void tri_copyModel(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN , int copyBoneStructures, int copyTextureData)
{
  //fprintf(stderr,MAGENTA "tri_copyModel ..\n" NORMAL);
  if (triModelOUT==0)           { return; }
  if (triModelIN==0)            { return; }
  if (triModelOUT==triModelIN)  { return; }

  tri_copyModelHeader( triModelOUT ,  triModelIN );

  unsigned int itemSize , count , allocationSize;

  if (triModelIN->header.nameSize!=0)
  {
   itemSize=sizeof(char); count=triModelIN->header.nameSize; allocationSize = itemSize * count;
   if (triModelOUT->name!=0)  { free(triModelOUT->name); triModelOUT->name=0; }
   if ((triModelIN->name!=0) && (allocationSize>0))  { triModelOUT->name = (char*) malloc(allocationSize+sizeof(char)); }
   memcpy(triModelOUT->name,triModelIN->name,allocationSize);
   triModelOUT->name[count]=0; //Null terminate
  }

  itemSize=sizeof(float); count=triModelIN->header.numberOfVertices; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of vertices ..\n", allocationSize);
  if (triModelOUT->vertices!=0)  { free(triModelOUT->vertices); triModelOUT->vertices=0; }
  if ((triModelIN->vertices!=0) && (allocationSize>0))  { triModelOUT->vertices = (float*) malloc(allocationSize); }
  memcpy(triModelOUT->vertices,triModelIN->vertices,allocationSize);

  itemSize=sizeof(float); count=triModelIN->header.numberOfNormals; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of normals ..\n", allocationSize);
  if (triModelOUT->normal!=0)  { free(triModelOUT->normal); triModelOUT->normal=0; }
  if ((triModelIN->normal!=0) && (allocationSize>0))  { triModelOUT->normal = (float*) malloc(allocationSize); }
  memcpy(triModelOUT->normal        , triModelIN->normal        , allocationSize);

  itemSize=sizeof(float); count=triModelIN->header.numberOfColors; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of colors ..\n", allocationSize);
  if (triModelOUT->colors!=0)  { free(triModelOUT->colors); triModelOUT->colors=0; }
  if ((triModelIN->colors!=0) && (allocationSize>0))  { triModelOUT->colors=(float*) malloc(allocationSize); }
  memcpy(triModelOUT->colors        , triModelIN->colors        , allocationSize);

  itemSize=sizeof(float); count=triModelIN->header.numberOfTextureCoords; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of textures ..\n", allocationSize);
  if (triModelOUT->textureCoords!=0)  { free(triModelOUT->textureCoords); triModelOUT->textureCoords=0; }
  if ((triModelIN->textureCoords!=0) && (allocationSize>0))  { triModelOUT->textureCoords=(float*) malloc(allocationSize); }
  memcpy(triModelOUT->textureCoords , triModelIN->textureCoords , allocationSize);

  itemSize=sizeof(unsigned int); count=triModelIN->header.numberOfIndices; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of indices ..\n", allocationSize);
  if (triModelOUT->indices!=0)  { free(triModelOUT->indices); triModelOUT->indices=0; }
  if ((triModelIN->indices!=0) && (allocationSize>0))  { triModelOUT->indices = (unsigned int*) malloc(allocationSize); }
  memcpy(triModelOUT->indices , triModelIN->indices       , allocationSize);


  if (triModelOUT->bones!=0)  { free(triModelOUT->bones); triModelOUT->bones=0; }

  if (copyBoneStructures)
  {
     tri_copyBones(triModelOUT,triModelIN);
  } else
  {
    fprintf(stderr,RED "tri_copyModel NOT copying bone structures..\n" NORMAL);
    if (triModelOUT->bones!=0)  { free(triModelOUT->bones); triModelOUT->bones=0; }
    triModelOUT->header.numberOfBones=0;
  }

  if (copyTextureData)
  {
    itemSize=sizeof(char); count=triModelIN->header.textureDataChannels * triModelIN->header.textureDataWidth * triModelIN->header.textureDataHeight ; allocationSize = itemSize * count;
    //fprintf(stderr,"Copying %u bytes of textureData ..\n", allocationSize);
    if (triModelOUT->textureData!=0)  { free(triModelOUT->textureData); triModelOUT->textureData=0; }
    if ((triModelIN->textureData!=0) && (allocationSize>0))  { triModelOUT->textureData = (unsigned char *) malloc(allocationSize); }
    memcpy(triModelOUT->textureData , triModelIN->textureData , allocationSize);
  }

 return ;
}

void copyModelTri(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN , int copyBoneStructures)
{
  fprintf(stderr,YELLOW "copyModelTri is deprecated\n" NORMAL);
  tri_copyModel(triModelOUT,triModelIN,copyBoneStructures,0);
}


int tri_loadModel(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Reading TRI model -> %s \n",filename );
  FILE *fd=0;
  fd = fopen(filename,"rb");
  if (fd!=0)
    {
        size_t n;
        unsigned int itemSize=0 , count=0;

        n = fread(&triModel->header , sizeof(struct TRI_Header), 1 , fd);
        if (n!= sizeof(struct TRI_Header)) { fprintf(stderr,"Incomplete read of TRI Header\n"); }

        if (triModel->header.floatSize!=sizeof(float))
             {
                 fprintf(stderr,"Size of float (%u/%lu) is different , cannot load \n",triModel->header.floatSize,sizeof(float));
                 fclose(fd);
                 return 0;
             }
        if (triModel->header.triType != TRI_LOADER_VERSION )
            {//TRI_LOADER_VERSION changes can lead to bugs let's have a HUGE warning message
             fprintf(stderr,RED " ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n" NORMAL);
              fprintf(stderr,YELLOW"  Incompatible triloader file , cannot load \n");
              if (triModel->header.triType>TRI_LOADER_VERSION )
                {
                  fprintf(stderr,"  You need to upgrade your TRI Loader code to be able to read this file \n");
                  fprintf(stderr,"  If you have included this reader in a tool, you need to \n");
                  fprintf(stderr,"  Copy && Paste the current version of model_loader_tri.c,model_loader_tri.h,model_loader_transform_joints.c and model_loader_transform_joints.h \n");
                  fprintf(stderr,"  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/ModelLoader\n");
                } else
              if (triModel->header.triType<TRI_LOADER_VERSION )
                {
                  fprintf(stderr,"  This is an old .TRI file-version that was dropped! \n");
                  fprintf(stderr,"  In order to keep the spec clean there is no backwards compatibility\n");
                  fprintf(stderr,"  I am sorry about this but there is no other way to keep things manageable\n");
                  fprintf(stderr,"  Hopefully the file format will stabilize to a version and will stop changing\n");
                  fprintf(stderr,"  If you really wish to open this file please revert to an older state of the repository\n");


                  //IF I EVER CHANGE THE VERSION AGAIN I SHOULD ALWAYS UPDATE LAST STABLE COMMIT HERE..
                  if (triModel->header.triType==8)
                  {
                      fprintf(stderr,"  The last stable commit that opens the file you want is 8efe78de3a6ee7f6c1a40c2bdccf91b1eee8d883\n");
                      fprintf(stderr,"  or https://github.com/AmmarkoV/RGBDAcquisition/commit/8efe78de3a6ee7f6c1a40c2bdccf91b1eee8d883\n\n\n");

                      fprintf(stderr,"  mkdir oldVersion && cd oldVersion\n");
                      fprintf(stderr,"  git clone https://github.com/AmmarkoV/RGBDAcquisition/ \n");
                      fprintf(stderr,"  cd RGBDAcquisition\n");
                      fprintf(stderr,"  git checkout 8efe78de3a6ee7f6c1a40c2bdccf91b1eee8d883\n\n\n\n");
                  }

                  fprintf(stderr,"  Thank you for your understanding\n");
                }
              fprintf(stderr,"   " NORMAL);
             fprintf(stderr,RED " ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n" NORMAL);

             fclose(fd);
             return 0;
            }


        //Write file name for internal usage..
        triModel->name = ( char * ) malloc ( sizeof(char) * (triModel->header.nameSize+1) );
        n = fread(triModel->name  , sizeof(char) , triModel->header.nameSize , fd);
        if (n!=sizeof(char) *triModel->header.nameSize) { fprintf(stderr,"Incomplete read of header name field\n"); }

        triModel->name[triModel->header.nameSize]=0;
        fprintf(stderr,"Internal name is -> %s \n",triModel->name);


        if (triModel->header.numberOfVertices)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfVertices;
         fprintf(stderr,"Reading %u bytes of vertex\n", itemSize * count );
         triModel->vertices = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->vertices , itemSize , count , fd);
         tri_warnIncompleteReads("vertices",count,n);
        } else {  fprintf(stderr,"No vertices specified \n"); }

        if (triModel->header.numberOfNormals)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfNormals;
         fprintf(stderr,"Reading %u bytes of normal\n", itemSize * count );
         triModel->normal = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->normal , itemSize , count , fd);
         tri_warnIncompleteReads("normals",count,n);
        } else {  fprintf(stderr,"No normals specified \n"); }


        if (triModel->header.numberOfTextureCoords)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Reading %u bytes of textures\n",itemSize * count);
         triModel->textureCoords = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->textureCoords , itemSize , count , fd);
         tri_warnIncompleteReads("textures",count,n);
        }  else {  fprintf(stderr,"No texture coords specified \n"); }

        if (triModel->header.numberOfColors)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfColors;
         fprintf(stderr,"Reading %u bytes of colors\n",itemSize * count);
         triModel->colors = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->colors ,  itemSize , count , fd);
         tri_warnIncompleteReads("colors",count,n);
        } else {  fprintf(stderr,"No colors specified \n"); }

        if (triModel->header.numberOfIndices)
        {
         itemSize=sizeof(unsigned int); count=triModel->header.numberOfIndices;
         fprintf(stderr,"Reading %u bytes of indices\n",itemSize * count);
         triModel->indices = ( unsigned int * ) malloc ( itemSize * count );
         n = fread(triModel->indices , itemSize , count , fd);
         tri_warnIncompleteReads("indices",count,n);
        } else {  fprintf(stderr,"No indices specified \n"); }

        if (triModel->header.numberOfBones)
        {
         fprintf(stderr,"Reading %u bones\n",triModel->header.numberOfBones);

         triModel->bones = (struct TRI_Bones *) malloc(sizeof(struct TRI_Bones) * triModel->header.numberOfBones);
         memset(triModel->bones, 0 , sizeof(struct TRI_Bones) * triModel->header.numberOfBones);


         if (triModel->bones)
         {
          unsigned int boneNum=0;
          for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
          {
          //First read dimensions of bone string and the number of weights for the bone..
          count=1;
          triModel->bones[boneNum].info = (struct TRI_Bones_Header*) malloc(sizeof(struct TRI_Bones_Header));
          memset( triModel->bones[boneNum].info , 0 , sizeof(struct TRI_Bones_Header) );
          n = fread(triModel->bones[boneNum].info , sizeof(struct TRI_Bones_Header), 1 , fd);
          tri_warnIncompleteReads("bone header",count,n);

          //Check value
          //fprintf(stderr,"Bone %u \n",boneNum);
          //print4x4DMatrixTRI("triModel->bones[boneNum].info->finalVertexTransformation", triModel->bones[boneNum].info->finalVertexTransformation);

          //Allocate enough space for the bone string , read it  , and null terminate it
          itemSize = sizeof(char);         count = triModel->bones[boneNum].info->boneNameSize;
          triModel->bones[boneNum].boneName = ( char * ) malloc ( (itemSize+2)*count );
          memset( triModel->bones[boneNum].boneName , 0 , (itemSize+2)*count );
          n = fread(triModel->bones[boneNum].boneName , itemSize , count , fd);
          tri_warnIncompleteReads("bone names",count,n);


          //Allocate enough space for the weight values , and read them
          itemSize = sizeof(float);        count = triModel->bones[boneNum].info->boneWeightsNumber;
          triModel->bones[boneNum].weightValue = ( float * ) malloc ( itemSize * count );
          memset( triModel->bones[boneNum].weightValue , 0 , itemSize * count );
          n = fread(triModel->bones[boneNum].weightValue , itemSize , count , fd);
          tri_warnIncompleteReads("bone weights",count,n);

          //Allocate enough space for the weight indexes , and read them
          itemSize = sizeof(unsigned int); count = triModel->bones[boneNum].info->boneWeightsNumber;
          triModel->bones[boneNum].weightIndex = ( unsigned int * ) malloc ( itemSize * count );
          memset( triModel->bones[boneNum].weightIndex , 0 , itemSize * count );
          n = fread(triModel->bones[boneNum].weightIndex , itemSize , count , fd);
          tri_warnIncompleteReads("bone weight indices",count,n);


          if (triModel->bones[boneNum].info->numberOfBoneChildren == 0 )
           {
            triModel->bones[boneNum].boneChild  = 0;
           } else
           {
            itemSize = sizeof(unsigned int); count = triModel->bones[boneNum].info->numberOfBoneChildren;
            triModel->bones[boneNum].boneChild  = ( unsigned int * ) malloc ( itemSize * count );
            memset( triModel->bones[boneNum].boneChild , 0 , itemSize * count );
            n = fread( triModel->bones[boneNum].boneChild  , itemSize , count , fd);
           }

          }
         }
        } else {  fprintf(stderr,"No bones specified \n"); }



        if ( triModel->header.textureDataWidth * triModel->header.textureDataHeight * triModel->header.textureDataChannels > 0 )
        {
         itemSize=sizeof(unsigned char); count = triModel->header.textureDataWidth * triModel->header.textureDataHeight * triModel->header.textureDataChannels;
         fprintf(stderr,"Reading %u bytes of texture data\n",itemSize * count);
         triModel->textureData = (unsigned char * ) malloc ( itemSize * count );
         n = fread(triModel->textureData , itemSize , count , fd);
         tri_warnIncompleteReads("texture data",count,n);
        } else {  fprintf(stderr,"No texture data specified \n"); }



        //printTRIBoneStructure(triModel, 1);
        printTRIModel(triModel);

        fclose(fd);

        return 1;
    } else
    {
       fprintf(stderr,"Failed reading file %s \n",filename);
    }
  return 0;
}



int loadModelTri(const char * filename , struct TRI_Model * triModel)
{
    fprintf(stderr,YELLOW " loadModelTri is deprecated \n" NORMAL);
    return tri_loadModel(filename,triModel);
}

int tri_findBone(struct TRI_Model * triModel,const char * searchName ,TRIBoneID * boneIDResult)
{
if ( (triModel->header.numberOfBones) && (triModel->bones!=0) && (boneIDResult!=0) )
  {
   //If the boneIDResult is set to something other than zero then do a "hail mary" check
   //avoiding the loop..
   if ( (*boneIDResult!=0) && (*boneIDResult<triModel->header.numberOfBones) )
   {
     if (strcmp(searchName,triModel->bones[*boneIDResult].boneName)==0)
            {
              //Fast Path, we knew the answer all along..
              //This might happen on subsequent calls to this function..
              return 1;
            }
   }

   TRIBoneID boneNum=0;
   for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
     {
        if ( strcmp(searchName,triModel->bones[boneNum].boneName)==0)
            {
              //fprintf(stderr,GREEN "Found bone %s ( %u ) \n" NORMAL , searchName , boneNum);
              *boneIDResult=boneNum;
              return 1;
            }
            // else { fprintf(stderr,RED "bone %s != %s\n" NORMAL,searchName,triModel->bones[boneNum].boneName); }
     }
  }

  //fprintf(stderr,RED "Could not find bone %s ( %u bones total ) \n" NORMAL , searchName , triModel->header.numberOfBones);
 return 0;
}


void tri_lowercase(char * str)
{
    char * a = str;
    if (a!=0)
    {
        while (*a!=0)
        {
            *a = tolower(*a);
            ++a;
        }
    }

    return;
}

void tri_removeunderscore(char * str)
{
    char * a = str;
    if (a!=0)
    {
        unsigned int l = strlen(str);
        if (l-2>0)
        {
            if (a[l-2]=='_')
            {
                a[l-2]='.';
            }
        }
    }
}



int tri_makeAllBoneNamesLowerCase(struct TRI_Model * triModel)
{
    if (triModel==0)
    {
        return 0;
    }
    //----------------------------------------
    for (unsigned int boneID=0; boneID<triModel->header.numberOfBones; boneID++)
    {
        tri_lowercase(triModel->bones[boneID].boneName);
    }

    return 1;
}


int tri_updateBoneName(struct TRI_Model * triModel,unsigned int boneID,const char * newBoneName)
{
    if (triModel == 0) { return 0; }
    if (newBoneName == 0) { return 0; }

    unsigned int newBoneNameSize = strlen(newBoneName);

    if (triModel->bones[boneID].info->boneNameSize+1 <= newBoneNameSize )
    {
      //We need more space to update the bone name!
      free(triModel->bones[boneID].boneName);
      triModel->bones[boneID].info->boneNameSize = newBoneNameSize+1; // + null terminator
      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (newBoneNameSize+1) );
      snprintf(triModel->bones[boneID].boneName,triModel->bones[boneID].info->boneNameSize,"%s",newBoneName);
    } else
    {
      //The bone name size is large enough to accomodate our new joint name..!
      snprintf(triModel->bones[boneID].boneName,triModel->bones[boneID].info->boneNameSize,"%s",newBoneName);
    }

    return 1;
}



int tri_removePrefixFromAllBoneNames(struct TRI_Model * triModel,const char * prefix)
{
    if (triModel==0)
    {
        return 0;
    }
    if (prefix==0)
    {
        return 0;
    }
    //----------------------------------------
    unsigned int prefixLength = strlen(prefix);

    for (TRIBoneID boneID=0; boneID<triModel->header.numberOfBones; boneID++)
    {
        char * boneName = triModel->bones[boneID].boneName;
        unsigned int fullBoneNameLength = strlen(boneName);

        if ( triModel->bones[boneID].boneName ==0 )
        {
            fprintf(stderr,"Invalid bone name encountered %u \n",boneID);
        }
        else
        {
            char * result = strstr(triModel->bones[boneID].boneName,prefix);
            if (result!=0)
            {
                snprintf(result,fullBoneNameLength,"%s",boneName+prefixLength);
            }
        }
    }

    return 1;
}


int tri_packageBoneDataPerVertex(struct TRI_BonesPackagedPerVertex * boneDataPerVertex,struct TRI_Model * model,int onlyGatherBoneTransforms)
{
 boneDataPerVertex->numberOfBones          = model->header.numberOfBones;
 boneDataPerVertex->numberOfBonesPerVertex = 4; //Maximum 4 bones per vertex, if you change this to 3 i.e. dont forget to change the shader!


 if (!onlyGatherBoneTransforms)
 {
  //Special gathering per vertex of all the bone stuff
  //===================================================================================================================================
  if (boneDataPerVertex->boneIndexes!=0)
      {
         free(boneDataPerVertex->boneIndexes);
         boneDataPerVertex->boneIndexes=0;
      }
  boneDataPerVertex->sizeOfBoneIndexes      = model->header.numberOfVertices * boneDataPerVertex->numberOfBonesPerVertex * sizeof(unsigned int);
  boneDataPerVertex->boneIndexes            = (unsigned int *) malloc(boneDataPerVertex->sizeOfBoneIndexes);
  memset(boneDataPerVertex->boneIndexes,0,boneDataPerVertex->sizeOfBoneIndexes); //Make sure empty bones are clean
  //===================================================================================================================================
  if (boneDataPerVertex->boneWeightValues!=0)
      {
         free(boneDataPerVertex->boneWeightValues);
         boneDataPerVertex->boneWeightValues=0;
      }
   boneDataPerVertex->sizeOfBoneWeightValues = model->header.numberOfVertices * boneDataPerVertex->numberOfBonesPerVertex * sizeof(float);
   boneDataPerVertex->boneWeightValues       = (float *) malloc(boneDataPerVertex->sizeOfBoneWeightValues);
   //It is really important bone weight values on unused bones are set to zero and delivered clean so that missing bones get zeroed out..
   memset(boneDataPerVertex->boneWeightValues,0,boneDataPerVertex->sizeOfBoneWeightValues); //Make sure empty bones are clean
   //===================================================================================================================================

   //We need to store per vertex A) the boneID B) the Weight!
   //This will get streamed to the shader to be enable the joints
      if ( (boneDataPerVertex->boneIndexes!=0) && (boneDataPerVertex->boneWeightValues!=0) )
      {
       //fprintf(stderr,"Will now pack %u bones..\n",model->header.numberOfBones);
       for (unsigned int boneID=0; boneID<model->header.numberOfBones; boneID++)
       {
        //fprintf(stderr,"Bone[%u]-> weights = %u \n",boneID,model->bones[boneID].info->boneWeightsNumber);
        for (unsigned int boneWeightID=0; boneWeightID<model->bones[boneID].info->boneWeightsNumber; boneWeightID++)
        {
         //V is the vertice we will be working in this loop
         unsigned int vertexID = model->bones[boneID].weightIndex[boneWeightID];
         //W is the weight that we have for the specific bone
         float boneWeightValue = model->bones[boneID].weightValue[boneWeightID];

         //Ok we now our target vertexID and the boneID and the boneWeightValue
         //We will try to fill in the specified index per vertex shader data with our new infos!
         char foundASpotToPutThisBoneData = 0;
         for (unsigned int vertexBoneSpot=0; vertexBoneSpot<boneDataPerVertex->numberOfBonesPerVertex; vertexBoneSpot++)
         {
           unsigned int bonePerVertexAddress = (vertexID*boneDataPerVertex->numberOfBonesPerVertex)+vertexBoneSpot;
           if ( (!foundASpotToPutThisBoneData) && (boneDataPerVertex->boneWeightValues[bonePerVertexAddress] == 0.0) )
           {
             //Great! we found a clean spot to put our data..!
             boneDataPerVertex->boneIndexes     [bonePerVertexAddress] = boneID;
             boneDataPerVertex->boneWeightValues[bonePerVertexAddress] = boneWeightValue;
             foundASpotToPutThisBoneData=1;
             break; // Break since we just found a clean spot..
             //BUG: There is a problem here, I am not sure if it is a logic error or GCC optimization
             //when the break is called this loop should stop executing however I needed to add the foundASpotToPutThisBoneData
             //in order to keep this from triggering multiple times..
           } //If no clean spot found data gets ignored!
         }
        }
       }
      }

 }




      //Special gathering of all the 4x4 matrixes
      if (boneDataPerVertex->boneTransforms!=0)
      {
         free(boneDataPerVertex->boneTransforms);
         boneDataPerVertex->boneTransforms=0;
      }
      boneDataPerVertex->sizeOfBoneTransforms   = (model->header.numberOfBones) * 16 * sizeof(float);
      boneDataPerVertex->boneTransforms         = (float *) malloc(boneDataPerVertex->sizeOfBoneTransforms);
      if (boneDataPerVertex->boneTransforms!=0)
      {
       for (unsigned int boneID=0; boneID<model->header.numberOfBones; boneID++)
         {
           unsigned int targetBoneTransformIndex = boneID*16;
           if ( model->bones[boneID].info!=0 )
           {
             float * s = model->bones[boneID].info->finalVertexTransformation;
             float * t = &boneDataPerVertex->boneTransforms[targetBoneTransformIndex];

             t[0]=s[0];    t[1]=s[1];    t[2]=s[2];    t[3]=s[3];
             t[4]=s[4];    t[5]=s[5];    t[6]=s[6];    t[7]=s[7];
             t[8]=s[8];    t[9]=s[9];    t[10]=s[10];  t[11]=s[11];
             t[12]=s[12];  t[13]=s[13];  t[14]=s[14];  t[15]=s[15];

             //NOTE: OpenGL expects a transposed matrix but we say to OGL to do it itself in uploadGeometry.c without wasting our CPU time here!
             //transpose4x4FMatrix(targetBoneTransformMatrix); //This could also be handled on the shader..!
           }
         }
      }

  return 1;
}






int findTRIBoneWithName(struct TRI_Model * triModel,const char * searchName ,TRIBoneID * boneIDResult)
{
    fprintf(stderr,YELLOW "findTRIBoneWithName is deprecated \n" NORMAL );
    return tri_findBone(triModel,searchName,boneIDResult);
}


int tri_dropAlphaFromTexture(struct TRI_Model * triModel)
{
    if (triModel!=0)
    {
        if (triModel->header.textureDataChannels==3)
        {
          //Already dropped alpha..
          return 1;
        }

        if (triModel->textureData!=0)
        {
          if (triModel->header.textureDataChannels==4)
           {
             unsigned char * trgPtr = triModel->textureData;
             unsigned char * srcPtr = triModel->textureData;
             unsigned char * srcPtrLimit = triModel->textureData + triModel->header.textureDataWidth * triModel->header.textureDataHeight * triModel->header.textureDataChannels;
             while (srcPtr<srcPtrLimit)
             {
               unsigned char r = *srcPtr; ++srcPtr;
               unsigned char g = *srcPtr; ++srcPtr;
               unsigned char b = *srcPtr; ++srcPtr;
               //char a = *srcPtr;
               ++srcPtr;
               //--------------------------
               *trgPtr = r; ++trgPtr;
               *trgPtr = g; ++trgPtr;
               *trgPtr = b; ++trgPtr;
               //Target won't have an alpha channel..!
             }

             triModel->header.textureDataChannels=3;
           }
        }
    }
    return 0;
}

int tri_packTextureInModel(struct TRI_Model * triModel,unsigned char * pixels , unsigned int width ,unsigned int height, unsigned int bitsperpixel , unsigned int channels)
{
  if (triModel!=0)
  {
    if (triModel->textureData!=0)
    {
        free(triModel->textureData);
        triModel->textureData=0;
    }

    triModel->header.textureDataWidth     = width;
    triModel->header.textureDataHeight    = height;
    triModel->header.textureDataChannels  = channels;
    triModel->header.textureUploadedToGPU = 0;
    triModel->header.textureBindGLBuffer  = 0;

    triModel->textureData = (unsigned char*) malloc(sizeof(unsigned char) * width * height * channels);

    if (triModel->textureData!=0)
    {
      memcpy(triModel->textureData,pixels,sizeof(unsigned char) * width * height * channels);
      return 1;
    } else
    {
     //Roll back texture settings since we failed packing it in..
     triModel->header.textureDataWidth    = 0;
     triModel->header.textureDataHeight   = 0;
     triModel->header.textureDataChannels = 0;
    }
  }

  return 0;
}


int tri_flipTexture(struct TRI_Model * triModel,char flipX,char flipY)
{
  if (triModel!=0)
  {
       if ( (triModel->textureCoords!=0) && (triModel->header.numberOfTextureCoords>0) )
       {
          float * t = triModel->textureCoords;
          float * tLimit = triModel->textureCoords + triModel->header.numberOfTextureCoords;

          while (t<tLimit)
          {
           float *u = t; ++t;
           float *v = t; ++t;

           if (flipX)
              { *u = 1.0 - *u; }
           if (flipY)
              { *v = 1.0 - *v; }
          }

          return 1;
       }
  }
 return 0;
}



int tri_paintModelUsingTexture(struct TRI_Model * triModel,unsigned char * pixels , unsigned int width ,unsigned int height, unsigned int bitsperpixel , unsigned int channels)
{
  if (triModel!=0)
  {
    if ( (triModel->header.numberOfColors>0 ) && ( triModel->header.numberOfTextureCoords>0) )
    {
       if ( (triModel->colors!=0) && (triModel->textureCoords!=0) )
       {
         float * clr = triModel->colors;
         float * clrLimit = triModel->colors + triModel->header.numberOfColors;

         float * tex = triModel->textureCoords;
         while (clr<clrLimit)
         {
           float u = *tex; ++tex;
           float v = *tex; ++tex;

           unsigned int x = (unsigned int) (u * width);
           unsigned int y = (unsigned int) (v * height);

           if ( (x<width) && (y<height) )
           {
            unsigned char * p = pixels + (y * width * channels) + x * channels;
            *clr = (float) *p/255; ++clr; ++p;
            *clr = (float) *p/255; ++clr; ++p;
            *clr = (float) *p/255; ++clr; ++p;
           } else
           {
             clr+=3;
           }
         }

         return 1;
       }
    }
  }
  fprintf(stderr,"Failed to paint TRI file..\n");
  return 0;
}


int tri_paintModel(struct TRI_Model * triModel,char r, char g, char b)
{
  if (triModel!=0)
  {
    if (triModel->header.numberOfColors>0 )
    {
       if (triModel->colors!=0)
       {
         float rP = (float) r/255;
         float gP = (float) g/255;
         float bP = (float) b/255;
         float * clr = triModel->colors;
         float * clrLimit = triModel->colors + triModel->header.numberOfColors;

         while (clr<clrLimit)
         {
           *clr = rP; ++clr;
           *clr = gP; ++clr;
           *clr = bP; ++clr;
         }

         return 1;
       }
    }
  }
  fprintf(stderr,"Failed to paint TRI file..\n");
  return 0;
}

int tri_saveModel(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Writing TRI model -> %s \n",filename );
  unsigned int itemSize=0 , count=0;

  FILE * fd = fopen(filename,"wb");
  if (fd!=0)
    {
        triModel->header.nameSize=strlen(filename);
        triModel->header.triType = TRI_LOADER_VERSION;
        triModel->header.floatSize =(unsigned int ) sizeof(float);
        triModel->header.TRIMagic[0] = 'T';
        triModel->header.TRIMagic[1] = 'R';
        triModel->header.TRIMagic[2] = 'I';
        triModel->header.TRIMagic[3] = '3';
        triModel->header.TRIMagic[4] = 'D';

        fwrite (&triModel->header , sizeof(struct TRI_Header), 1 , fd);


        //Write file name for internal usage..
        fwrite (filename, sizeof(char), triModel->header.nameSize , fd);

        if ( (triModel->header.numberOfVertices) && (triModel->vertices!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfVertices;
         fprintf(stderr,"Writing %u bytes of vertices\n",itemSize*count);
         fwrite (triModel->vertices , itemSize , count , fd);
        }

        if ( (triModel->header.numberOfNormals) && (triModel->normal!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfNormals;
         fprintf(stderr,"Writing %u bytes of normal\n",itemSize*count);
         fwrite (triModel->normal , itemSize , count  , fd);
        }

        if ( (triModel->header.numberOfTextureCoords) && (triModel->textureCoords!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Writing %u bytes of textureCoords\n",itemSize*count);
         fwrite (triModel->textureCoords,itemSize , count,fd);
        }

        if ( (triModel->header.numberOfColors) && (triModel->colors!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfColors;
         fprintf(stderr,"Writing %u bytes of colors\n",itemSize*count);
         fwrite (triModel->colors , itemSize , count, fd);
        }

        if ( (triModel->header.numberOfIndices) && (triModel->indices!=0) )
        {
         itemSize=sizeof(unsigned int); count=triModel->header.numberOfIndices;
         fprintf(stderr,"Writing %u bytes of indices\n",itemSize*count);
         fwrite (triModel->indices ,itemSize , count, fd);
        }


        if ( (triModel->header.numberOfBones) && (triModel->bones!=0) )
        {
         struct TRI_Bones_Header emptyBonesHeader={0};
         unsigned int boneNum=0;
         fprintf(stderr,"Writing %u bones\n",triModel->header.numberOfBones);
         for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
         {
          //fprintf(stderr,"%u\n",boneNum);

          if (triModel->bones[boneNum].info!=0)
           { fwrite ( triModel->bones[boneNum].info     , sizeof(struct TRI_Bones_Header) , 1 , fd); } else
           { fwrite ( &emptyBonesHeader                 , sizeof(struct TRI_Bones_Header) , 1 , fd); }

          fwrite ( triModel->bones[boneNum].boneName    , sizeof(char)                    , triModel->bones[boneNum].info->boneNameSize      , fd);
          fwrite ( triModel->bones[boneNum].weightValue , sizeof(float)                   , triModel->bones[boneNum].info->boneWeightsNumber , fd);
          fwrite ( triModel->bones[boneNum].weightIndex , sizeof(unsigned int)            , triModel->bones[boneNum].info->boneWeightsNumber , fd);

          if ( (triModel->bones[boneNum].info->numberOfBoneChildren>0) && (triModel->bones[boneNum].boneChild) )
           { fwrite ( triModel->bones[boneNum].boneChild , sizeof(unsigned int)     , triModel->bones[boneNum].info->numberOfBoneChildren , fd); }
         }
        }


        if ( (triModel->header.textureDataWidth * triModel->header.textureDataHeight * triModel->header.textureDataChannels > 0) && (triModel->textureData!=0) )
        {
         itemSize=sizeof(char);  count = triModel->header.textureDataWidth * triModel->header.textureDataHeight * triModel->header.textureDataChannels;
         fprintf(stderr,"Writing %u bytes of texture data\n",itemSize*count);
         fwrite (triModel->textureData ,itemSize , count, fd);
        }


        fflush(fd);
        fclose(fd);
        fprintf(stderr,GREEN "Success writing TRI model %s to disk \n" NORMAL,filename);
        return 1;
    }
  return 0;
}


int saveModelTri(const char * filename , struct TRI_Model * triModel)
{
    fprintf(stderr,YELLOW "Deprecated call saveModelTri \n");
    return tri_saveModel(filename,triModel);
}


int tri_simpleMergeOfTRIInContainer(struct TRI_Model * triModel,struct TRI_Container * container)
{
        triModel->header.nameSize=7;
        triModel->header.triType = TRI_LOADER_VERSION;
        triModel->header.floatSize =(unsigned int ) sizeof(float);
        triModel->header.TRIMagic[0] = 'T';
        triModel->header.TRIMagic[1] = 'R';
        triModel->header.TRIMagic[2] = 'I';
        triModel->header.TRIMagic[3] = '3';
        triModel->header.TRIMagic[4] = 'D';
        triModel->name = (char * ) malloc(sizeof(char) * (triModel->header.nameSize+1));
        snprintf(triModel->name,triModel->header.nameSize,"merged");

        triModel->header.numberOfVertices = 0;
        triModel->header.numberOfNormals = 0;
        triModel->header.numberOfTextureCoords = 0;
        triModel->header.numberOfColors = 0;
        triModel->header.numberOfIndices = 0;
        triModel->header.numberOfBones = 0;
        for (unsigned int meshID = 0; meshID < container->header.numberOfMeshes; meshID++ )
        {
          triModel->header.numberOfVertices      += container->mesh[meshID].header.numberOfVertices;
          triModel->header.numberOfNormals       += container->mesh[meshID].header.numberOfNormals;
          triModel->header.numberOfTextureCoords += container->mesh[meshID].header.numberOfTextureCoords;
          triModel->header.numberOfColors        += container->mesh[meshID].header.numberOfColors;
          triModel->header.numberOfIndices       += container->mesh[meshID].header.numberOfIndices;
          //Ignore Bones..
        }


        unsigned int itemSize=0 , count=0;
        //------------------------------------------------------------------
        if (triModel->header.numberOfVertices)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfVertices;
         fprintf(stderr,"Allocating %u bytes of vertex data\n", itemSize * count );
         triModel->vertices = ( float * ) malloc ( itemSize * count );
        }

        if (triModel->header.numberOfNormals)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfNormals;
         fprintf(stderr,"Allocating %u bytes of normal data\n", itemSize * count );
         triModel->normal = ( float * ) malloc ( itemSize * count );
        }

        if (triModel->header.numberOfTextureCoords)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Allocating %u bytes of texture data\n",itemSize * count);
         triModel->textureCoords = ( float * ) malloc ( itemSize * count );
        }

        if (triModel->header.numberOfColors)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfColors;
         fprintf(stderr,"Allocating %u bytes of colors\n",itemSize * count);
         triModel->colors = ( float * ) malloc ( itemSize * count );
        }

        if (triModel->header.numberOfIndices)
        {
         itemSize=sizeof(unsigned int); count=triModel->header.numberOfIndices;
         fprintf(stderr,"Allocating %u bytes of indices\n",itemSize * count);
         triModel->indices = ( unsigned int * ) malloc ( itemSize * count );
        }
        //------------------------------------------------------------------

        unsigned int numberOfVertices = 0;
        unsigned int numberOfNormals = 0;
        unsigned int numberOfTextureCoords = 0;
        unsigned int numberOfColors = 0;
        unsigned int numberOfIndices = 0;
        //unsigned int numberOfBones = 0;
        for (unsigned int meshID = 0; meshID < container->header.numberOfMeshes; meshID++ )
        {
          //Append Vertices..
          if (container->mesh[meshID].vertices!=0)
          {
           memcpy( triModel->vertices + numberOfVertices , container->mesh[meshID].vertices , container->mesh[meshID].header.numberOfVertices * sizeof(float) );
           numberOfVertices+=container->mesh[meshID].header.numberOfVertices;
          }

          //Append Normals..
          if (container->mesh[meshID].normal!=0)
          {
           memcpy( triModel->normal + numberOfNormals , container->mesh[meshID].normal , container->mesh[meshID].header.numberOfNormals * sizeof(float) );
           numberOfNormals+=container->mesh[meshID].header.numberOfNormals;
          }

          //Append Texture Coords..
          if (container->mesh[meshID].textureCoords!=0)
          {
           memcpy( triModel->textureCoords + numberOfTextureCoords , container->mesh[meshID].textureCoords , container->mesh[meshID].header.numberOfTextureCoords * sizeof(float) );
           numberOfTextureCoords+=container->mesh[meshID].header.numberOfTextureCoords;
          }

          //Append Colors..
          if (container->mesh[meshID].colors!=0)
          {
           memcpy( triModel->colors + numberOfColors , container->mesh[meshID].colors , container->mesh[meshID].header.numberOfColors * sizeof(float) );
           numberOfColors+=container->mesh[meshID].header.numberOfColors;
          }

          //Append Indices..
          if (container->mesh[meshID].indices!=0)
          {
           memcpy( triModel->indices + numberOfIndices , container->mesh[meshID].indices , container->mesh[meshID].header.numberOfIndices * sizeof(unsigned int) );
           numberOfIndices+=container->mesh[meshID].header.numberOfIndices;
          }
          //Ignore Bones..
        }




    return 1;
}



struct pakHeader
{
  char id[4];
  int offset;
  int size;
};

struct pakFile
{
  char name[56];
  int offset;
  int size;
};

// Based on the code here :  https://quakewiki.org/wiki/.pak
/* pak_filename : the os filename of the .pak file */
/* filename     : the name of the file you're trying to load from the .pak file (remember to use forward slashes for the path) */
/* out_filesize : if not null, the loaded file's size will be returned here */
/* returns a malloced buffer containing the file contents (remember to free it later), or NULL if any error occurred */
void *pak_load_file(const char *pak_filename, const char *filename, int *out_filesize)
{
  FILE *fp;
  struct pakHeader pak_header;
  int num_files;
  int i;
  struct pakFile pak_file;
  void *buffer;

  fp = fopen(pak_filename, "rb");
  if (!fp)
    return NULL;

  if (!fread(&pak_header, sizeof(pak_header), 1, fp)) { fclose(fp); return 0; }

  if (memcmp(pak_header.id, "PACK", 4) != 0)          { fclose(fp); return 0; }

  //Do conversion from little endian
  //pak_header.offset = LittleLong(pak_header.offset);
  //pak_header.size = LittleLong(pak_header.size);

  num_files = pak_header.size / sizeof(struct pakFile);

  if (fseek(fp, pak_header.offset, SEEK_SET) != 0)    { fclose(fp); return 0; }

  for (i = 0; i < num_files; i++)
  {
    if (!fread(&pak_file, sizeof(struct pakFile), 1, fp)) { fclose(fp); return 0; }

    if (!strcmp(pak_file.name, filename))
    {
      //Do conversion from little endian
      //pak_file.offset = LittleLong(pak_file.offset);
      //pak_file.size = LittleLong(pak_file.size);

      if (fseek(fp, pak_file.offset, SEEK_SET) != 0) { fclose(fp); return 0; }

      buffer = malloc(pak_file.size);
      if (!buffer) { return 0; }

      if (!fread(buffer, pak_file.size, 1, fp))
      {
        free(buffer);
        fclose(fp);
        return 0;
      }

      if (out_filesize)
        { *out_filesize = pak_file.size; }
      return buffer;
    }
  }

  fclose(fp);
  return 0;
}





//#define INCLUDE_OPENGL_CODE 1
void doTriDrawCalllist(struct TRI_Model * tri )
{
 #if INCLUDE_OPENGL_CODE
  unsigned int i=0,z;


  glBegin(GL_TRIANGLES);
    if (tri->header.numberOfIndices > 0 )
    {
     //fprintf(stderr,MAGENTA "drawing indexed TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
     unsigned int faceTriA,faceTriB,faceTriC,faceTriA_X,faceTriA_Y,faceTriA_Z,faceTriB_X,faceTriB_Y,faceTriB_Z,faceTriC_X,faceTriC_Y,faceTriC_Z;

     for (i = 0; i < tri->header.numberOfIndices/3; i++)
     {
      faceTriA = tri->indices[(i*3)+0];      faceTriB = tri->indices[(i*3)+1];      faceTriC = tri->indices[(i*3)+2];
      faceTriA_X = (faceTriA*3)+0;           faceTriA_Y = (faceTriA*3)+1;           faceTriA_Z = (faceTriA*3)+2;
      faceTriB_X = (faceTriB*3)+0;           faceTriB_Y = (faceTriB*3)+1;           faceTriB_Z = (faceTriB*3)+2;
      faceTriC_X = (faceTriC*3)+0;           faceTriC_Y = (faceTriC*3)+1;           faceTriC_Z = (faceTriC*3)+2;

      if (tri->normal)   { glNormal3f(tri->normal[faceTriA_X],tri->normal[faceTriA_Y],tri->normal[faceTriA_Z]); }
      if ( tri->colors ) { glColor3f(tri->colors[faceTriA_X],tri->colors[faceTriA_Y],tri->colors[faceTriA_Z]);  }
      glVertex3f(tri->vertices[faceTriA_X],tri->vertices[faceTriA_Y],tri->vertices[faceTriA_Z]);

      if (tri->normal)   { glNormal3f(tri->normal[faceTriB_X],tri->normal[faceTriB_Y],tri->normal[faceTriB_Z]); }
      if ( tri->colors ) { glColor3f(tri->colors[faceTriB_X],tri->colors[faceTriB_Y],tri->colors[faceTriB_Z]);  }
      glVertex3f(tri->vertices[faceTriB_X],tri->vertices[faceTriB_Y],tri->vertices[faceTriB_Z]);

      if (tri->normal)   { glNormal3f(tri->normal[faceTriC_X],tri->normal[faceTriC_Y],tri->normal[faceTriC_Z]); }
      if ( tri->colors ) { glColor3f(tri->colors[faceTriC_X],tri->colors[faceTriC_Y],tri->colors[faceTriC_Z]);  }
      glVertex3f(tri->vertices[faceTriC_X],tri->vertices[faceTriC_Y],tri->vertices[faceTriC_Z]);
	 }
    } else
    {
      //fprintf(stderr,BLUE "drawing flat TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
      for (i=0; i<tri->header.numberOfVertices/3; i++)
        {
         z=(i*3)*3;
         if (tri->normal) { glNormal3f(tri->normal[z+0],tri->normal[z+1],tri->normal[z+2]); }
         if (tri->colors) { glColor3f(tri->colors[z+0],tri->colors[z+1],tri->colors[z+2]);  }
                            glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);

         z+=3;
         if (tri->normal) { glNormal3f(tri->normal[z+0],tri->normal[z+1],tri->normal[z+2]); }
         if (tri->colors) { glColor3f(tri->colors[z+0],tri->colors[z+1],tri->colors[z+2]);  }
                            glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);
         z+=3;
         if (tri->normal) { glNormal3f(tri->normal[z+0],tri->normal[z+1],tri->normal[z+2]); }
         if (tri->colors) { glColor3f(tri->colors[z+0],tri->colors[z+1],tri->colors[z+2]);  }
                            glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);
        }
    }
  glEnd();
 #else
  fprintf(stderr,YELLOW "OpenGL code not compiled in this model loader TRI code.. \n" NORMAL);
 #endif // INCLUDE_OPENGL_CODE
}



