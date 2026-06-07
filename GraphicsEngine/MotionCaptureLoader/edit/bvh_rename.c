#include "bvh_rename.h"


#include "../mathLibrary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void lowercase(char *str)
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

void uppercase(char * str)
{
  char * a = str;
  if (a!=0)
        {
          while (*a!=0)
            {
            *a = toupper(*a);
            ++a;
            }
        }
 return;
}










void bvh_setLimbFlags(struct BVH_MotionCapture * bvhMotion)
{
    if (bvhMotion==0)
        {
            fprintf(stderr,"Cannot set limb flags joints of NULL bvh file\n");
            return ;
        }
    if (bvhMotion->jointHierarchy==0)
        {
            fprintf(stderr,"Cannot set limb flag of NULL bvh hierarchy \n");
            return ;
        }
    if (bvhMotion->jointHierarchySize==0)
        {
            fprintf(stderr,"Cannot set limb flag of empty bvh hierarchy\n");
            return ;
        }

    BVHJointID jID;
    for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
        {
            unsigned int pJID=bvhMotion->jointHierarchy[jID].parentJoint;
            char * jPN = 0;
            if (pJID<bvhMotion->jointHierarchySize)
                {
                    jPN = bvhMotion->jointHierarchy[pJID].jointName;
                }
            char * jN = bvhMotion->jointHierarchy[jID].jointName;

            if ( (jN!=0) && (jPN!=0) )
                {
                    //----------------------------------------------------------------
                    if (
                        (strcmp(jN,"abdomen")==0) || (strcmp(jPN,"abdomen")==0) || (strcmp(jN,"chest")==0) || (strcmp(jPN,"chest")==0) || (strcmp(jN,"neck")==0) || (strcmp(jPN,"neck")==0)
                        || (strcmp(jN,"hip")==0) || (strcmp(jPN,"hip")==0)
                    )
                        {
                            bvhMotion->jointHierarchy[jID].partOfHierarchy.isAPartOfTorso=1;
                        }
                    else
                        //----------------------------------------------------------------
                        if (
                            (strcmp(jN,"neck")==0) || (strcmp(jPN,"neck")==0) || (strcmp(jN,"nose")==0) || (strcmp(jPN,"nose")==0) || (strcmp(jN,"head")==0) || (strcmp(jPN,"head")==0)
                        )
                            {
                                bvhMotion->jointHierarchy[jID].partOfHierarchy.isAPartOfHead=1;
                            }
                        else
                            //----------------------------------------------------------------
                            if (
                                (strcmp(jN,"rshoulder")==0) || (strcmp(jPN,"rshoulder")==0) || (strcmp(jN,"relbow")==0) || (strcmp(jPN,"relbow")==0) || (strcmp(jN,"rhand")==0)
                            )
                                {
                                    bvhMotion->jointHierarchy[jID].partOfHierarchy.isAPartOfRightArm=1;
                                }
                    //----------------------------------------------------------------
                    if (
                        (strcmp(jN,"lshoulder")==0) || (strcmp(jPN,"lshoulder")==0) || (strcmp(jN,"lelbow")==0) || (strcmp(jPN,"lelbow")==0) || (strcmp(jN,"lhand")==0)
                    )
                        {
                            bvhMotion->jointHierarchy[jID].partOfHierarchy.isAPartOfLeftArm=1;
                        }
                    //----------------------------------------------------------------
                    if (
                        (strcmp(jN,"rhip")==0) || (strcmp(jN,"rknee")==0) || (strcmp(jN,"rfoot")==0) || (strcmp(jPN,"rfoot")==0)
                    )
                        {
                            bvhMotion->jointHierarchy[jID].partOfHierarchy.isAPartOfRightFoot=1;
                        }
                    //----------------------------------------------------------------
                    if (
                        (strcmp(jN,"lhip")==0) || (strcmp(jN,"lknee")==0) || (strcmp(jN,"lfoot")==0) || (strcmp(jPN,"lfoot")==0)
                    )
                        {
                            bvhMotion->jointHierarchy[jID].partOfHierarchy.isAPartOfLeftFoot=1;
                        }
                    //----------------------------------------------------------------


                }
        }
}


void bvh_setTorsoImmunityForJoints(struct BVH_MotionCapture * bvhMotion)
{
    BVHJointID jID;
    for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
        {
            char * jN = bvhMotion->jointHierarchy[jID].jointName;
            if (jN!=0)
                {
                    if (
                        (strcmp(jN,"abdomen")==0) ||
                        (strcmp(jN,"chest")==0) ||
                        (strcmp(jN,"neck")==0) ||
                        (strcmp(jN,"lcollar")==0) ||
                        (strcmp(jN,"rcollar")==0) ||
                        (strcmp(jN,"lshoulder")==0) ||
                        (strcmp(jN,"rshoulder")==0) ||
                        (strcmp(jN,"rbuttock")==0) ||
                        (strcmp(jN,"rhip")==0) ||
                        (strcmp(jN,"lbuttock")==0) ||
                        (strcmp(jN,"lhip")==0) ||
                        (strcmp(jN,"hip")==0)
                    )
                        {
                            bvhMotion->jointHierarchy[jID].isImmuneToTorsoOcclusions=1;
                        }
                    else
                        {
                            bvhMotion->jointHierarchy[jID].isImmuneToTorsoOcclusions=0;
                        }
                }
        }
}



/*! djb2
This algorithm (k=33) was first reported by dan bernstein many years ago in comp.lang.c.
another version of this algorithm (now favored by bernstein) uses xor: hash(i) = hash(i - 1) * 33 ^ str[i];
the magic of number 33 (why it works better than many other constants, prime or not) has never been adequately explained.
Needless to say , this is our hash function..!
*/
unsigned long hashFunctionJoints(const char *str)
{
    if (str==0) return 0;
    if (str[0]==0) return 0;

    unsigned long hash = 5381; //<- magic
    int c=1;

    while (c != 0)
        {
            c = *str++;
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        }

    return hash;
}



void bvh_updateJointNameHashes(struct BVH_MotionCapture * bvhMotion)
{
    BVHJointID jID=0;
    for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
        {
            memcpy(bvhMotion->jointHierarchy[jID].jointNameLowercase,bvhMotion->jointHierarchy[jID].jointName,MAX_BVH_JOINT_NAME+1);
            //snprintf(bvhMotion->jointHierarchy[jID].jointNameLowercase,MAX_BVH_JOINT_NAME,"%s",bvhMotion->jointHierarchy[jID].jointName);
            lowercase(bvhMotion->jointHierarchy[jID].jointNameLowercase);
            bvhMotion->jointHierarchy[jID].jointNameHash = hashFunctionJoints(bvhMotion->jointHierarchy[jID].jointNameLowercase);
        }

}

void bvh_renameJointsToLowercase(struct BVH_MotionCapture * bvhMotion)
{
   BVHJointID jID=0;
   for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
        {
            lowercase(bvhMotion->jointHierarchy[jID].jointName);
        }
   bvh_updateJointNameHashes(bvhMotion);
}


void bvh_renameJointsForCompatibility(struct BVH_MotionCapture * bvhMotion)
{
    if (bvhMotion==0)
        {
            fprintf(stderr,"Cannot rename joints of NULL bvh file\n");
            return ;
        }
    if (bvhMotion->jointHierarchy==0)
        {
            fprintf(stderr,"Cannot rename joints of NULL bvh hierarchy \n");
            return ;
        }
    if (bvhMotion->jointHierarchySize==0)
        {
            fprintf(stderr,"Cannot rename joints of empty bvh hierarchy\n");
            return ;
        }

    unsigned int totalChangesPerformed = 0;
    BVHJointID jID=0;


    for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
        {
            char * jOr = bvhMotion->jointHierarchy[jID].jointName;
            char * jN  = bvhMotion->jointHierarchy[jID].jointNameLowercase;
            if (jN!=0)
                {
                    //This is already lowercase
                    //lowercase(jN);


                    int changedSomething = 0;

                    //-------------------------------------------------------------------------------------------------
                    //------------------------------------------TORSO--------------------------------------------------
                    //-------------------------------------------------------------------------------------------------
                    if  ( (strcmp(jN,"hip")==0) || (strcmp(jN,"hips")==0) )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"hip");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"hip");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"abdomen")==0) || (strcmp(jN,"spine")==0) || (strcmp(jN,"torso")==0)  || (strcmp(jN,"waist")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"abdomen");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"abdomen");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"chest")==0) || (strcmp(jN,"chest2")==0) || (strcmp(jN,"spine1")==0)  || (strcmp(jN,"spine2")==0) || (strcmp(jN,"torso2")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"chest");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"chest");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------


                    //-------------------------------------------------------------------------------------------------
                    //---------------------------------------LEFT FOOT------------------------------------------------
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"lefthip")==0) || (strcmp(jN,"leftupleg")==0) || (strcmp(jN,"lthigh")==0)  || (strcmp(jN,"leftupperLeg")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lhip");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lhip");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"leftknee")==0) || (strcmp(jN,"leftlowleg")==0) || (strcmp(jN,"leftleg")==0)  || (strcmp(jN,"lshin")==0) || (strcmp(jN,"leftlowerleg")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lknee");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lknee");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"leftankle")==0) || (strcmp(jN,"leftfoot")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lfoot");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lfoot");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------


                    //-------------------------------------------------------------------------------------------------
                    //---------------------------------------RIGHT FOOT------------------------------------------------
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"righthip")==0) || (strcmp(jN,"rightupleg")==0) || (strcmp(jN,"rthigh")==0)  || (strcmp(jN,"rightupperleg")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"rhip");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"rhip");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"rightknee")==0) || (strcmp(jN,"rightlowleg")==0) || (strcmp(jN,"rightleg")==0)  || (strcmp(jN,"rshin")==0) || (strcmp(jN,"rightlowerleg")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"rknee");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"rknee");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"rightankle")==0) || (strcmp(jN,"rightfoot")==0) || (strcmp(jN,"rfoot")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"rfoot");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"rfoot");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------


                    //-------------------------------------------------------------------------------------------------
                    //---------------------------------------LEFT HAND------------------------------------------------
                    //-------------------------------------------------------------------------------------------------
                    if  (
                         (strcmp(jN,"leftcollar")==0) || (strcmp(jN,"lcollar")==0) || (strcmp(jN,"leftshoulder")==0)  || (strcmp(jN,"lcolr")==0)
                        )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lcollar");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lcollar");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                         (strcmp(jN,"leftuparm")==0) || (strcmp(jN,"leftarm")==0) || (strcmp(jN,"lshldr")==0)  || (strcmp(jN,"leftshoulder")==0) || (strcmp(jN,"leftupperarm")==0)
                        )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lshoulder");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lshoulder");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                         (strcmp(jN,"leftelbow")==0) || (strcmp(jN,"leftlowarm")==0) || (strcmp(jN,"leftforearm")==0)  || (strcmp(jN,"lforearm")==0) || (strcmp(jN,"leftlowerarm")==0)
                        )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lelbow");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lelbow");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"leftwrist")==0) || (strcmp(jN,"lefthand")==0) || (strcmp(jN,"lhand")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"lhand");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"lhand");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }


                    //-------------------------------------------------------------------------------------------------
                    //---------------------------------------RIGHT HAND------------------------------------------------
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"rightcollar")==0) || (strcmp(jN,"rcollar")==0) || (strcmp(jN,"rightshoulder")==0)  || (strcmp(jN,"rcolr")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"rcollar");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"rcollar");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"rightuparm")==0) || (strcmp(jN,"rightarm")==0) || (strcmp(jN,"rshldr")==0)  || (strcmp(jN,"rightshoulder")==0) || (strcmp(jN,"rightupperarm")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"rshoulder");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"rshoulder");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"rightelbow")==0) || (strcmp(jN,"rightlowarm")==0) || (strcmp(jN,"rightforearm")==0)  || (strcmp(jN,"rforearm")==0) || (strcmp(jN,"rightlowerarm")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"relbow");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"relbow");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------
                    if  (
                        (strcmp(jN,"rightwrist")==0) || (strcmp(jN,"righthand")==0) || (strcmp(jN,"rhand")==0)
                    )
                        {
                            snprintf(jN,MAX_BVH_JOINT_NAME,"rhand");
                            snprintf(jOr,MAX_BVH_JOINT_NAME,"rhand");
                            changedSomething=1;
                            ++totalChangesPerformed;
                        }
                    //-------------------------------------------------------------------------------------------------

                    if (changedSomething)
                        {
                            //Since jN is pointing to the lower case buffer and we have corrected the lowercase buffer lets copy this back to the regular joint name!
                            //snprintf(bvhMotion->jointHierarchy[jID].jointName,MAX_BVH_JOINT_NAME,"%s",bvhMotion->jointHierarchy[jID].jointNameLowercase);
                            memcpy(bvhMotion->jointHierarchy[jID].jointNameLowercase,bvhMotion->jointHierarchy[jID].jointName,MAX_BVH_JOINT_NAME+1);
                        }

                }
        }

  if (totalChangesPerformed>0)
  {
    fprintf(stderr,"Renamed %u joints for easier compatibility with different armature names..\n",totalChangesPerformed);
  }

    bvh_setLimbFlags(bvhMotion);
    bvh_setTorsoImmunityForJoints(bvhMotion);
    bvh_updateJointNameHashes(bvhMotion);
}
