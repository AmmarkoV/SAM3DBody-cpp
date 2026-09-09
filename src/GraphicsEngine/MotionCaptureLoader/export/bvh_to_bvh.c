#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_bvh.h"

#include "../calculate/bvh_project.h"


void indent(FILE * fp , unsigned int indentation)
{
  unsigned int i=0;
  for (i=0; i<indentation*2; i++)
  {
   fprintf(fp," ");
  }
}


int valueIsZero(float val)
{
  if ((-0.00001<val) && (val<0.00001) ) { return 1; }
  return 0;
}

void writeBVHHierarchyClosingSection(
                                     FILE * fp ,
                                     struct BVH_MotionCapture * mc,
                                     unsigned int hierarchyLevelStart,
                                     unsigned int hierarchyLevelEnd
                                   )
{
  unsigned int hierarchyLevel=0;

  //fprintf(stderr,"Close Sections from %u->%u : ",hierarchyLevelEnd,hierarchyLevelStart);
  for (hierarchyLevel=hierarchyLevelEnd+1; hierarchyLevel>hierarchyLevelStart; hierarchyLevel--)
  {
   //fprintf(stderr,"%u ",hierarchyLevel);
   indent(fp,hierarchyLevel-1);  fprintf(fp,"}\n");
  }

  //fprintf(stderr,"done \n");
}


int writeBVHHierarchyOpenningSection(
                                     FILE * fp ,
                                     struct BVH_MotionCapture * mc,
                                     BVHJointID jID
                                    )
{
  unsigned int in=mc->jointHierarchy[jID].hierarchyLevel;


   if (mc->jointHierarchy[jID].isRoot)
        { indent(fp,in); fprintf(fp,"ROOT %s\n",mc->jointHierarchy[jID].jointName); }
        else
   if (mc->jointHierarchy[jID].isEndSite)
        { indent(fp,in); fprintf(fp,"End Site\n"); }
        else
        { indent(fp,in); fprintf(fp,"JOINT %s\n",mc->jointHierarchy[jID].jointName); }

  indent(fp,in); fprintf(fp,"{\n");

  ++in;
  indent(fp,in);
  //Write offset usign the least possible filesize
  fprintf(fp,"OFFSET");
  if (mc->jointHierarchy[jID].offset[0]!=0.0) { fprintf(fp," %f",mc->jointHierarchy[jID].offset[0]);   } else
                                              { fprintf(fp," 0");   }
  if (mc->jointHierarchy[jID].offset[1]!=0.0) { fprintf(fp," %f",mc->jointHierarchy[jID].offset[1]);   } else
                                              { fprintf(fp," 0");   }
  if (mc->jointHierarchy[jID].offset[2]!=0.0) { fprintf(fp," %f\n",mc->jointHierarchy[jID].offset[2]); } else
                                              { fprintf(fp," 0\n"); }



  if (!mc->jointHierarchy[jID].isEndSite)
  {//----------------------------------------------------------------------------------------------
   indent(fp,in);
   fprintf(
           fp,"CHANNELS %u",
           mc->jointHierarchy[jID].loadedChannels
          );

   unsigned int channelID=0;
   for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
   {
     fprintf(fp," %s",channelNames[(unsigned int)mc->jointHierarchy[jID].channelType[channelID]]);
   }
   fprintf(fp,"\n");

  }//----------------------------------------------------------------------------------------------

 return 1;
}



int dumpBVHToBVH(
                  const char * bvhFilename,
                  struct BVH_MotionCapture * mc,
                  int writeHierarchy,
                  int writeMotion
                )
{
   BVHJointID rootJID;

   if (!bvh_getRootJointID(mc,&rootJID)) { return 0;}

   FILE * fp = fopen(bvhFilename,"w");
   if (fp!=0)
   {

     if (writeHierarchy)
     {
     fprintf(fp,"HIERARCHY\n");
     //--------------------------------------------------------------------------------------
     unsigned int nextHierarchyLevel; // This gets always overwritten.. = 0;
     unsigned int hasNext;

     for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
        {
          unsigned int currentHierarchyLevel = mc->jointHierarchy[jID].hierarchyLevel;

          if (jID+1<mc->jointHierarchySize) { hasNext=1; } else { hasNext = 0; }

          if (hasNext) { nextHierarchyLevel=mc->jointHierarchy[jID+1].hierarchyLevel; } else
                       { nextHierarchyLevel=0; }

          writeBVHHierarchyOpenningSection(fp,mc,jID);

          if (nextHierarchyLevel < currentHierarchyLevel)
          {
              writeBVHHierarchyClosingSection(
                                            fp ,
                                            mc,
                                            nextHierarchyLevel,
                                            currentHierarchyLevel
                                          );
          }
        }
      }


     if (writeMotion)
     {
      fprintf(fp,"MOTION\n");
      //--------------------------------------------------------------------------------------
      fprintf(fp,"Frames: %u\n",mc->numberOfFrames);
      fprintf(fp,"Frame Time: %0.8f\n",mc->frameTime);

      for (unsigned int fID=0; fID<mc->numberOfFrames; fID++)
      {
        for (unsigned int mID=fID*mc->numberOfValuesPerFrame; mID<(fID+1)*mc->numberOfValuesPerFrame; mID++)
         {
           if (valueIsZero(mc->motionValues[mID]))
           { //Make file smaller and cleaner instead of spamming it with values like 4.8946335e-17
            fprintf(fp,"0 ");
           } else
           {
            fprintf(fp,"%f ",mc->motionValues[mID]);
           }
         }
        fprintf(fp,"\n");
      }
     }
     fclose(fp);
     return 1;
   }

 return 0;
}

