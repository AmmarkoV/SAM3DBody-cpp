#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_tri_pose.h"

#include "../../TrajectoryParser/InputParser_C.h"


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

unsigned int bvh_resolveBVHToTRIJoint(struct bvhToTRI * bvhtri,const char * jName)
{
  unsigned int jID=0;

  for (jID=0; jID<bvhtri->numberOfJointAssociations; jID++)
  {
    if (strcmp(jName,bvhtri->jointAssociation[jID].bvhJointName)==0)
    {
      return jID;
    }
  }
 return 0;
}

char bvh_setRotationName(char name)
{
 switch (name)
   {
     case 'x' :
     case 'X' :
       return BVH_ROTATION_X;
     break;

     case 'y' :
     case 'Y' :
       return BVH_ROTATION_Y;
     break;

     case 'z' :
     case 'Z' :
       return BVH_ROTATION_Z;
     break;
   }
 return 0;
}


void bvh_offsetLabelsToSigns(
                              struct bvhToTRI * bvhtri,
                              unsigned int jID,
                              char *oX,
                              char *oY,
                              char *oZ
                            )
{
  char * jName = bvhtri->jointAssociation[jID].bvhJointName;
  fprintf(stderr,"RotationOrderWithSign %s(%s,%s,%s)\n",jName,oX,oY,oZ);

  bvhtri->jointAssociation[jID].rotationOrder[0].sign=1.0;
  if (oX[0]=='+') { ++oX; } else
  if (oX[0]=='-') { ++oX; bvhtri->jointAssociation[jID].rotationOrder[0].sign=-1.0; }

  bvhtri->jointAssociation[jID].rotationOrder[1].sign=1.0;
  if (oY[0]=='+') { ++oY; } else
  if (oY[0]=='-') { ++oY; bvhtri->jointAssociation[jID].rotationOrder[1].sign=-1.0; }

  bvhtri->jointAssociation[jID].rotationOrder[2].sign=1.0;
  if (oZ[0]=='+') { ++oZ; } else
  if (oZ[0]=='-') { ++oZ; bvhtri->jointAssociation[jID].rotationOrder[2].sign=-1.0; }

  fprintf(
           stderr,"RotationSign %s(%0.2f,%0.2f,%0.2f)\n",jName,
           bvhtri->jointAssociation[jID].rotationOrder[0].sign,
           bvhtri->jointAssociation[jID].rotationOrder[1].sign,
           bvhtri->jointAssociation[jID].rotationOrder[2].sign
         );

   bvhtri->jointAssociation[jID].rotationOrder[0].rotID = bvh_setRotationName(*oX);
   bvhtri->jointAssociation[jID].rotationOrder[1].rotID = bvh_setRotationName(*oY);
   bvhtri->jointAssociation[jID].rotationOrder[2].rotID = bvh_setRotationName(*oZ);

  fprintf(stderr,"RotationOrder %s(%s,%s,%s)\n",jName,oX,oY,oZ);
}




int parseJointAssociation(
                           struct bvhToTRI * bvhtri,
                           struct BVH_MotionCapture * mc,
                           const char * originalName,
                           const char * associatedName
                         )
{
 unsigned int jID=bvhtri->numberOfJointAssociations;
 unsigned int realJID=0;

 if ( bvh_getJointIDFromJointNameNocase(mc,originalName,&realJID) )
    {
           fprintf(stderr,"Associating #%u `%s` => `%s` \n", jID, originalName, associatedName);
           //------------------------------------------------------
           bvhtri->jointAssociation[jID].offset[0]=0.0;
           bvhtri->jointAssociation[jID].offset[1]=0.0;
           bvhtri->jointAssociation[jID].offset[2]=0.0;
           //------------------------------------------------------
           bvhtri->jointAssociation[jID].rotationOrder[0].sign=-1.0;
           bvhtri->jointAssociation[jID].rotationOrder[1].sign=-1.0;
           bvhtri->jointAssociation[jID].rotationOrder[2].sign=-1.0;
           //-------------------------------------------------------

           switch (mc->jointHierarchy[realJID].channelRotationOrder)
           {
             case  BVH_ROTATION_ORDER_XYZ :
               bvhtri->jointAssociation[jID].rotationOrder[0].rotID=BVH_ROTATION_X;
               bvhtri->jointAssociation[jID].rotationOrder[1].rotID=BVH_ROTATION_Y;
               bvhtri->jointAssociation[jID].rotationOrder[2].rotID=BVH_ROTATION_Z;
             break;
             case  BVH_ROTATION_ORDER_XZY :
               bvhtri->jointAssociation[jID].rotationOrder[0].rotID=BVH_ROTATION_X;
               bvhtri->jointAssociation[jID].rotationOrder[1].rotID=BVH_ROTATION_Z;
               bvhtri->jointAssociation[jID].rotationOrder[2].rotID=BVH_ROTATION_Y;
             break;
             case  BVH_ROTATION_ORDER_YXZ :
               bvhtri->jointAssociation[jID].rotationOrder[0].rotID=BVH_ROTATION_Y;
               bvhtri->jointAssociation[jID].rotationOrder[1].rotID=BVH_ROTATION_X;
               bvhtri->jointAssociation[jID].rotationOrder[2].rotID=BVH_ROTATION_Z;
             break;
             case  BVH_ROTATION_ORDER_YZX :
               bvhtri->jointAssociation[jID].rotationOrder[0].rotID=BVH_ROTATION_Y;
               bvhtri->jointAssociation[jID].rotationOrder[1].rotID=BVH_ROTATION_Z;
               bvhtri->jointAssociation[jID].rotationOrder[2].rotID=BVH_ROTATION_X;
             break;
             case  BVH_ROTATION_ORDER_ZXY :
               bvhtri->jointAssociation[jID].rotationOrder[0].rotID=BVH_ROTATION_Z;
               bvhtri->jointAssociation[jID].rotationOrder[1].rotID=BVH_ROTATION_X;
               bvhtri->jointAssociation[jID].rotationOrder[2].rotID=BVH_ROTATION_Y;
             break;
             case  BVH_ROTATION_ORDER_ZYX :
               bvhtri->jointAssociation[jID].rotationOrder[0].rotID=BVH_ROTATION_Z;
               bvhtri->jointAssociation[jID].rotationOrder[1].rotID=BVH_ROTATION_Y;
               bvhtri->jointAssociation[jID].rotationOrder[2].rotID=BVH_ROTATION_X;
             break;
             default :
               fprintf(stderr,"parseJointAssociation: incorrect rotation order %u\n",(unsigned int) mc->jointHierarchy[realJID].channelRotationOrder);
             break;
           }
           //-------------------------------------------------------
           bvhtri->jointAssociation[jID].useJoint=1;
           ++bvhtri->numberOfJointAssociations;
      return 1;
    }


 fprintf(stderr,RED "Failed creating association rule #%u `%s` => `%s` \n" NORMAL,jID,originalName,associatedName);
 return 0;
}











int bvh_loadBVHToTRIAssociationFile(
                                    const char * filename ,
                                    struct bvhToTRI * bvhtri,
                                    struct BVH_MotionCapture * mc
                                   )
{
  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
   struct InputParserC * ipc = InputParser_Create(4096,7);
   InputParser_SetDelimeter(ipc,0,' ');
   InputParser_SetDelimeter(ipc,1,'(');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,')');
   InputParser_SetDelimeter(ipc,4,'\t');
   InputParser_SetDelimeter(ipc,5,10);
   InputParser_SetDelimeter(ipc,6,13);


    ssize_t read;
    char * line = NULL;
    size_t len = 0;

    char nameA[512]={0};

    bvhtri->numberOfJointAssociations=0;
    unsigned int jID=0;

    while  ((read = getline(&line, &len, fp)) != -1)
    {
       int num = InputParser_SeperateWords(ipc,line,1);

       if (num>0)
       {
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ASSOCIATION"))
         {
           InputParser_GetWord(ipc,1,bvhtri->jointAssociation[jID].bvhJointName,MAX_BVH_JOINT_NAME);
           InputParser_GetWord(ipc,2,bvhtri->jointAssociation[jID].triJointName,MAX_BVH_JOINT_NAME);
           parseJointAssociation(
                                  bvhtri,
                                  mc,
                                  bvhtri->jointAssociation[jID].bvhJointName,
                                  bvhtri->jointAssociation[jID].triJointName
                                );
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ASSOCIATION_SAME"))
         {
           InputParser_GetWord(ipc,1,bvhtri->jointAssociation[jID].bvhJointName,MAX_BVH_JOINT_NAME);
           InputParser_GetWord(ipc,1,bvhtri->jointAssociation[jID].triJointName,MAX_BVH_JOINT_NAME);
           parseJointAssociation(
                                  bvhtri,
                                  mc,
                                  bvhtri->jointAssociation[jID].bvhJointName,
                                  bvhtri->jointAssociation[jID].triJointName
                                );
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ROTATION_ORDER"))
         {
           InputParser_GetWord(ipc,1,nameA,512);
           jID = bvh_resolveBVHToTRIJoint(bvhtri,nameA);
           InputParser_GetWord(ipc,2,bvhtri->jointAssociation[jID].rotationOrder[0].label,64);
           InputParser_GetWord(ipc,3,bvhtri->jointAssociation[jID].rotationOrder[1].label,64);
           InputParser_GetWord(ipc,4,bvhtri->jointAssociation[jID].rotationOrder[2].label,64);

           bvh_offsetLabelsToSigns(
                                   bvhtri,
                                   jID,
                                   bvhtri->jointAssociation[jID].rotationOrder[0].label,
                                   bvhtri->jointAssociation[jID].rotationOrder[1].label,
                                   bvhtri->jointAssociation[jID].rotationOrder[2].label
                                  );
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_OFFSET"))
        {
          InputParser_GetWord(ipc,1,nameA,512);
          jID = bvh_resolveBVHToTRIJoint(bvhtri,nameA);
          bvhtri->jointAssociation[jID].offset[0] = InputParser_GetWordFloat(ipc,2);
          bvhtri->jointAssociation[jID].offset[1] = InputParser_GetWordFloat(ipc,3);
          bvhtri->jointAssociation[jID].offset[2] = InputParser_GetWordFloat(ipc,4);
          fprintf(
                  stderr,"Offset %s #%u (%0.2f,%0.2f,%0.2f)\n",nameA,jID,
                  bvhtri->jointAssociation[jID].offset[0],
                  bvhtri->jointAssociation[jID].offset[1],
                  bvhtri->jointAssociation[jID].offset[2]
                 );
        } else
         if (InputParser_WordCompareAuto(ipc,0,"JOINT_SIGN"))
        {
          InputParser_GetWord(ipc,1,nameA,512);
          jID = bvh_resolveBVHToTRIJoint(bvhtri,nameA);
          bvhtri->jointAssociation[jID].rotationOrder[0].sign = InputParser_GetWordFloat(ipc,2);
          bvhtri->jointAssociation[jID].rotationOrder[1].sign = InputParser_GetWordFloat(ipc,3);
          bvhtri->jointAssociation[jID].rotationOrder[2].sign = InputParser_GetWordFloat(ipc,4);
          fprintf(
                  stderr,"Sign %s #%u (%0.2f,%0.2f,%0.2f)\n",nameA,jID,
                  bvhtri->jointAssociation[jID].rotationOrder[0].sign,
                  bvhtri->jointAssociation[jID].rotationOrder[1].sign,
                  bvhtri->jointAssociation[jID].rotationOrder[2].sign
                 );
        }
       }
    }

    fprintf( stderr,"%u Associations\n", bvhtri->numberOfJointAssociations );


    if (line!=0) { free(line); }

    fclose(fp);
    InputParser_Destroy(ipc);
    return 1;
  }
return 0;
}
