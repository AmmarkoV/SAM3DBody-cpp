#include "fromBVH.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <io.h>
#define read _read
#define write _write
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

static ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    if (lineptr == NULL || n == NULL || stream == NULL) return -1;
    if (*lineptr == NULL || *n == 0) {
        *n = 128;
        *lineptr = (char *)malloc(*n);
        if (*lineptr == NULL) return -1;
    }
    ssize_t count = 0;
    int ch;
    while ((ch = fgetc(stream)) != EOF) {
        if (count + 1 >= (ssize_t)*n) {
            size_t new_n = *n * 2;
            char *new_ptr = (char *)realloc(*lineptr, new_n);
            if (new_ptr == NULL) return -1;
            *lineptr = new_ptr;
            *n = new_n;
        }
        (*lineptr)[count++] = (char)ch;
        if (ch == '\n') break;
    }
    if (count == 0) return -1;
    (*lineptr)[count] = '\0';
    return count;
}
#else
#include <unistd.h>
#endif

#include <string.h>
#include "../../TrajectoryParser/InputParser_C.h"
#include "../edit/bvh_rename.h"

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



int getParentJoint(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint , unsigned int hierarchyLevel , unsigned int * parentJoint)
{
  if (currentJoint>=bvhMotion->MAX_jointHierarchySize)
  {
    fprintf(stderr,"getParentJoint: Incorrect currentJoint value\n");
    return 0;
  }

  if (hierarchyLevel==0)
  {
   //Already at root level..
   //No Parent joint..
   *parentJoint = 0;
   return 1;
  }

  unsigned int i = currentJoint;

  while (i>0)
  {
    if (bvhMotion->jointHierarchy[i].hierarchyLevel==hierarchyLevel-1)
    {
      //Found Parent Joint..!
      *parentJoint = i;
      return 1;
    }
    --i;
  }


  //We did not find something better than the root joint..
  *parentJoint = 0;
  return 1;
  //return 0;
}

int enumerateInputParserChannel(struct InputParserC * ipc , unsigned int argumentNumber)
{
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Xrodrigues") ) {return BVH_RODRIGUES_X; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Yrodrigues") ) {return BVH_RODRIGUES_Y; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Zrodrigues") ) {return BVH_RODRIGUES_Z; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Xrotation") )  {return BVH_ROTATION_X;  } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Yrotation") )  {return BVH_ROTATION_Y;  } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Zrotation") )  {return BVH_ROTATION_Z;  } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Wrotation") )  {return BVH_ROTATION_W;  } else //QBVH
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Xposition") )  {return BVH_POSITION_X;  } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Yposition") )  {return BVH_POSITION_Y;  } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Zposition") )  {return BVH_POSITION_Z;  }

  return BVH_CHANNEL_NONE;
}



int thisLineOnlyHasX(const char * line,const char x)
{
  unsigned int i=0;
  for (i=0; i<strlen(line); i++)
  {
    if (line[i]==' ')  { } else
    if (line[i]=='\t') { } else
    if (line[i]==10)   { } else
    if (line[i]==13)   { } else
    if (line[i]==x)    { } else
        {
           //fprintf(stderr,"Line Char %u is %u(%c)\n",i,(unsigned int) line[i],line[i]);
           return 0;
        }
  }
 return 1;
}


int pushNewBVHMotionState(struct BVH_MotionCapture * bvhMotion ,const char * parameters)
{
   if (bvhMotion==0)  { return 0; }
   if (parameters==0)  { return 0; }

   if (
         (bvhMotion->motionValues==0) ||
         (bvhMotion->motionValuesSize==0)
      )
   {
     fprintf(stderr,"Cannot pushNewBVHMotionState without space to store new information\n");
     return 0;
   }

   struct InputParserC * ipc = InputParser_Create(MAX_BVH_FILE_LINE_SIZE,4);
   if (ipc==0) { return 0; }

   InputParser_SetDelimeter(ipc,0,' ');
   InputParser_SetDelimeter(ipc,1,'\t');
   InputParser_SetDelimeter(ipc,2,10);
   InputParser_SetDelimeter(ipc,3,13);

   //unsigned int i=0;
   unsigned int numberOfParameters = InputParser_SeperateWordsCC(ipc,parameters,1);
   //fprintf(stderr,"MOTION command (%s) has %u parameters\n",parameters,numberOfParameters);


   if (numberOfParameters==bvhMotion->numberOfValuesPerFrame)
   {
     unsigned int finalMemoryLocation = (numberOfParameters-1) + (bvhMotion->numberOfFramesEncountered * bvhMotion->numberOfValuesPerFrame);
     if (finalMemoryLocation < bvhMotion->motionValuesSize)
     {
      /*
      fprintf(stderr,
              "Filling from %u to %u \n",
              bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame,
              numberOfParameters+bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame
             );*/

      for (unsigned int i=0; i<numberOfParameters; i++)
      {
        //fprintf(stderr,"P%u=%0.2f ",i,InputParser_GetWordFloat(ipc,i));
        unsigned int thisMemoryLocation = i+(bvhMotion->numberOfFramesEncountered * bvhMotion->numberOfValuesPerFrame);
        bvhMotion->motionValues[thisMemoryLocation] = InputParser_GetWordFloat(ipc,i);
      }
     }
     bvhMotion->numberOfFramesEncountered++;
   } else
   {
    //Unexpected input..
    fprintf(stderr,"Motion Expected had %u parameters we received %u\n",bvhMotion->numberOfValuesPerFrame,numberOfParameters);
    fprintf(stderr,"Unexpected line num (%u)  :\n" , bvhMotion->linesParsed);
    fprintf(stderr,"%s\n", parameters);

    fprintf(stderr,"For reference a motion vector should look like : ");
    for (unsigned int i=0; i<bvhMotion->numberOfValuesPerFrame; i++)
    {
       fprintf(stderr,"0 ");
    }

    fprintf(stderr,"\n");
    //exit(0);
   }

   InputParser_Destroy(ipc);
   return 1;
}


void errorReachedMemoryLimit(unsigned int jID)
{
  fprintf(stderr,RED "We have reached the maximum allowed number of joints in hierarchy (%u out of bounds)\n",jID);
  fprintf(stderr,    "don't know how this happened since dynamic allocation should cover any input size..\n" NORMAL);
  exit(1);
}



int fastBVHFileToDetermineNumberOfJointsAndMotionFields(struct BVH_MotionCapture * bvhMotion , FILE * fd)
{
  if (fd!=0)
  {
   //The outputs of this function are these
   bvhMotion->motionValuesSize=0;
   bvhMotion->MAX_jointHierarchySize = 0;
   //--------------------------------------

   struct InputParserC * ipc = InputParser_Create(4096,5);
   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,']');
   InputParser_SetDelimeter(ipc,3,10);
   InputParser_SetDelimeter(ipc,4,13);

   struct InputParserC * ipcB = InputParser_Create(4096,4);
   InputParser_SetDelimeter(ipcB,0,' ');
   InputParser_SetDelimeter(ipcB,1,'\t');
   InputParser_SetDelimeter(ipcB,2,10);
   InputParser_SetDelimeter(ipcB,3,13);

    unsigned int atHeaderSection=0;
    unsigned int jointsEncountered=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int hierarchyLevel=0;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    int done=0;
    while  ( (!done) && ((read = getline(&line, &len, fd)) != -1) )
    {
       ++bvhMotion->linesParsed;
       //printf("Retrieved line of length %zu :\n %s", read,line);
       int num = InputParser_SeperateWords(ipc,line,1);


      if (num>0)
      { //We have content..
       if (!atHeaderSection)
       {
          //We won't start parsing unless we reach a HIERARCHY line
          if (InputParser_WordCompareAuto(ipc,0,"HIERARCHY"))  { atHeaderSection=1; }
       } else
       {
         int numB = InputParser_SeperateWords(ipcB,line,1);
         if (numB>0)
         {
         if (InputParser_WordCompareAuto(ipcB,0,"ROOT"))
              {
               //---------------------------------------------------------------------------------------------------------------------
               //------------------------------------------------- ROOT --------------------------------------------------------------
               //---------------------------------------------------------------------------------------------------------------------
               ++jointsEncountered;
               //---------------------------------------------------------------------------------------------------------------------
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"JOINT"))
              {
               //---------------------------------------------------------------------------------------------------------------------
               //------------------------------------------------- JOINT -------------------------------------------------------------
               //---------------------------------------------------------------------------------------------------------------------
                ++jointsEncountered;
               //---------------------------------------------------------------------------------------------------------------------
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"End"))
             {
              //We encountered something like |End Site|
              if (InputParser_WordCompareAuto(ipcB,1,"Site"))
                   {
                    //---------------------------------------------------------------------------------------------------------------------
                    //------------------------------------------------- END SITE ----------------------------------------------------------
                    //---------------------------------------------------------------------------------------------------------------------
                     ++jointsEncountered;
                    //---------------------------------------------------------------------------------------------------------------------
                   }
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"CHANNELS"))
             {
                  //---------------------------------------------------------------------------------------------------------------------
                  //------------------------------------------------- CHANNELS ----------------------------------------------------------
                  //---------------------------------------------------------------------------------------------------------------------
                  //Reached something like  |CHANNELS 3 Zrotation Xrotation Yrotation| declaration
                  unsigned int loadedChannels = InputParser_GetWordInt(ipcB,1);
                  bvhMotion->motionValuesSize += loadedChannels;
                  //---------------------------------------------------------------------------------------------------------------------
             } else
         if ( (InputParser_WordCompareAuto(ipcB,0,"{")) || (thisLineOnlyHasX(line,'{')) )
             {
               ++hierarchyLevel;
             } else
         if ( (InputParser_WordCompareAuto(ipcB,0,"}")) || (thisLineOnlyHasX(line,'}') ) )
             {
              //We reached an } so we pop one hierarchyLevel
              if (hierarchyLevel>0)
                 {
                  --hierarchyLevel;
                 } else
                 {
                   fprintf(stderr,RED "Erroneous BVH hierarchy..\n" NORMAL);
                 }

              if (hierarchyLevel==0)
                 {
                  //We are done..
                  done=1;
                 }
              //-------------------------------------
             }
         } // We have header content
       } // We are at header section
      } //We have content
    } //We have line input from file

   //Free incoming line buffer..
   if (line) { free(line); line=0;}

   if (hierarchyLevel!=0)
   {
     atHeaderSection = 0; //We finished with header..
     fprintf(stderr,RED "Missing } braces , atHeaderSection->%u..\n" NORMAL,atHeaderSection);
   }

   //Remember max hierarchy size
   bvhMotion->MAX_jointHierarchySize = jointsEncountered;

   InputParser_Destroy(ipc);
   InputParser_Destroy(ipcB);
   //We need file to be open..!
   rewind(fd);
   bvhMotion->linesParsed=0; //reset lines parsed..
   return ( (bvhMotion->jointHierarchySize>0) &&  (bvhMotion->numberOfValuesPerFrame>0) );
  }
 return 0;
}



void bvh_populateStaticTransformationOfJoint(struct BVH_MotionCapture * bvhMotion,BVHJointID jID)
{
    float * m = bvhMotion->jointHierarchy[jID].staticTransformation.m;
    m[0] =1.0;    m[1] =0.0;   m[2] =0.0;    m[3] = (float) bvhMotion->jointHierarchy[jID].offset[0];
    m[4] =0.0;    m[5] =1.0;   m[6] =0.0;    m[7] = (float) bvhMotion->jointHierarchy[jID].offset[1];
    m[8] =0.0;    m[9] =0.0;   m[10]=1.0;    m[11]= (float) bvhMotion->jointHierarchy[jID].offset[2];
    m[12]=0.0;    m[13]=0.0;   m[14]=0.0;    m[15]=1.0;
}




int readBVHHeader(struct BVH_MotionCapture * bvhMotion,FILE * fd)
{
  fastBVHFileToDetermineNumberOfJointsAndMotionFields(bvhMotion,fd);
  fprintf(stderr,GREEN "Fast BVH parser revealed %u joints / %u motion fields\n" NORMAL,bvhMotion->MAX_jointHierarchySize,bvhMotion->numberOfValuesPerFrame);

  //Make space for an off by one mistake ;P
  //bvhMotion->MAX_jointHierarchySize+=1;
  //bvhMotion->motionValuesSize+=1;

  bvhMotion->linesParsed=0; //make sure these are clean..
  bvhMotion->numberOfValuesPerFrame = 0; //make sure these are clean..

  if  (bvhMotion->jointHierarchy!=0)       { free(bvhMotion->jointHierarchy);      bvhMotion->jointHierarchy=0;}
  bvhMotion->jointHierarchy       = (struct BVH_Joint *)                     malloc( sizeof(struct BVH_Joint) * bvhMotion->MAX_jointHierarchySize);

  if  (bvhMotion->jointToMotionLookup!=0)  { free(bvhMotion->jointToMotionLookup); bvhMotion->jointToMotionLookup=0;}
  bvhMotion->jointToMotionLookup  = (struct BVH_JointToMotion_LookupTable *) malloc( sizeof(struct BVH_JointToMotion_LookupTable) * bvhMotion->MAX_jointHierarchySize);

  if  (bvhMotion->motionToJointLookup!=0)  { free(bvhMotion->motionToJointLookup); bvhMotion->motionToJointLookup=0;}
  bvhMotion->motionToJointLookup  = (struct BVH_MotionToJoint_LookupTable *) malloc( sizeof(struct BVH_MotionToJoint_LookupTable) * bvhMotion->motionValuesSize); // bvhMotion->MAX_jointHierarchySize * 7

  if (
       (bvhMotion->jointHierarchy==0) || (bvhMotion->jointToMotionLookup==0) || (bvhMotion->motionToJointLookup==0)
     )
       {
         fprintf(stderr,RED "Failed allocating memory, giving up on loading BVH file\n" NORMAL);
         if  (bvhMotion->jointHierarchy!=0)       { free(bvhMotion->jointHierarchy);      bvhMotion->jointHierarchy=0;}
         if  (bvhMotion->jointToMotionLookup!=0)  { free(bvhMotion->jointToMotionLookup); bvhMotion->jointToMotionLookup=0;}
         if  (bvhMotion->motionToJointLookup!=0)  { free(bvhMotion->motionToJointLookup); bvhMotion->motionToJointLookup=0;}
         //----------------------------------
         bvhMotion->MAX_jointHierarchySize=0;
         bvhMotion->motionValuesSize=0;
         return 0;
       }

  //Clean everything..!
  memset(bvhMotion->jointHierarchy,0,sizeof(struct BVH_Joint) * bvhMotion->MAX_jointHierarchySize);
  memset(bvhMotion->jointToMotionLookup,0,sizeof(struct BVH_JointToMotion_LookupTable) * bvhMotion->MAX_jointHierarchySize);
  memset(bvhMotion->motionToJointLookup,0,sizeof(struct BVH_MotionToJoint_LookupTable) * bvhMotion->motionValuesSize);

  //Let's start parsing again..
  bvhMotion->linesParsed=0;

  int atHeaderSection=0;


  int debug=bvhMotion->debug;

  if (fd!=0)
  {
   struct InputParserC * ipc = InputParser_Create(4096,5);
   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,']');
   InputParser_SetDelimeter(ipc,3,10);
   InputParser_SetDelimeter(ipc,4,13);

   struct InputParserC * ipcB = InputParser_Create(4096,4);
   InputParser_SetDelimeter(ipcB,0,' ');
   InputParser_SetDelimeter(ipcB,1,'\t');
   InputParser_SetDelimeter(ipcB,2,10);
   InputParser_SetDelimeter(ipcB,3,13);

    unsigned int jNum=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int currentJoint=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int hierarchyLevel=0;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    int done=0;
    while  ( (!done) && ((read = getline(&line, &len, fd)) != -1) )
    {
       ++bvhMotion->linesParsed;
       //printf("Retrieved line of length %zu :\n %s", read,line);
       int numberOfTokensInBVHFileLineA = InputParser_SeperateWords(ipc,line,1);


      if (numberOfTokensInBVHFileLineA>0)
      { //We have content..
       if (!atHeaderSection)
       {
          //We won't start parsing unless we reach a HIERARCHY line
          if (InputParser_WordCompareAuto(ipc,0,"HIERARCHY"))  { atHeaderSection=1; }
       } else
       {
         int numberOfTokensInBVHFileLineB = InputParser_SeperateWords(ipcB,line,1);
         if (numberOfTokensInBVHFileLineB>0)
         {
         if (InputParser_WordCompareAuto(ipcB,0,"ROOT"))
              {
               //---------------------------------------------------------------------------------------------------------------------
               //------------------------------------------------- ROOT --------------------------------------------------------------
               //---------------------------------------------------------------------------------------------------------------------
               //We encountered something like |ROOT Hips|
               if (debug) {fprintf(stderr,"-R-");}
               //Store new ROOT Joint Name
               InputParser_GetWord(ipcB,1,bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME);

                //Also store lowercase version of joint name for internal use
                memcpy(bvhMotion->jointHierarchy[jNum].jointNameLowercase,bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME+1);
                //snprintf(bvhMotion->jointHierarchy[jNum].jointNameLowercase,MAX_BVH_JOINT_NAME,"%s",bvhMotion->jointHierarchy[jNum].jointName);
                lowercase(bvhMotion->jointHierarchy[jNum].jointNameLowercase);
                bvhMotion->jointHierarchy[jNum].jointNameHash = hashFunctionJoints(bvhMotion->jointHierarchy[jNum].jointNameLowercase);

               if (debug) {fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[jNum].jointName);}
               //Store new Joint Hierarchy Level
               //Rest of the information will be filled in when we reach an {


               if (jNum<bvhMotion->MAX_jointHierarchySize)
               {
                 bvhMotion->rootJointID=jNum;
                 bvhMotion->jointHierarchy[jNum].hierarchyLevel = hierarchyLevel;
                 bvhMotion->jointHierarchy[jNum].isRoot=1;
                 //Update lookup table to remember ordering
                 bvhMotion->jointToMotionLookup[jNum].jointMotionOffset  = bvhMotion->numberOfValuesPerFrame;
                 currentJoint=jNum;
                 ++jNum;
               } else
               {
                 fprintf(stderr,"We have run out of space for our joint hierarchy..\n");
                 errorReachedMemoryLimit(jNum);
               }
              //---------------------------------------------------------------------------------------------------------------------
              //---------------------------------------------------------------------------------------------------------------------
              //---------------------------------------------------------------------------------------------------------------------
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"JOINT"))
              {
               //---------------------------------------------------------------------------------------------------------------------
               //------------------------------------------------- JOINT -------------------------------------------------------------
               //---------------------------------------------------------------------------------------------------------------------

               //We encountered something like |JOINT Chest|
               if (debug) {fprintf(stderr,"-J-");}


               if (jNum<bvhMotion->MAX_jointHierarchySize)
               {
                //Store new Joint Name
                InputParser_GetWord(ipcB,1,bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME);

                //Also store lowercase version of joint name for internal use
                memcpy(bvhMotion->jointHierarchy[jNum].jointNameLowercase,bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME+1);
                //snprintf(bvhMotion->jointHierarchy[jNum].jointNameLowercase,MAX_BVH_JOINT_NAME,"%s",bvhMotion->jointHierarchy[jNum].jointName);
                lowercase(bvhMotion->jointHierarchy[jNum].jointNameLowercase);
                bvhMotion->jointHierarchy[jNum].jointNameHash = hashFunctionJoints(bvhMotion->jointHierarchy[jNum].jointNameLowercase);

                if (debug) {fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[jNum].jointName);}
                //Store new Joint Hierarchy Level
                //Rest of the information will be filled in when we reach an {
                bvhMotion->jointHierarchy[jNum].hierarchyLevel = hierarchyLevel;
                //Update lookup table to remember ordering
                bvhMotion->jointToMotionLookup[jNum].jointMotionOffset  = bvhMotion->numberOfValuesPerFrame;
                currentJoint=jNum;

                //---------------------------------------------------------------------------------------------------------------------
                //---------------------------------------------------------------------------------------------------------------------
                //---------------------------------------------------------------------------------------------------------------------
                ++jNum;
               }  else
               {
                 errorReachedMemoryLimit(jNum);
               }
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"End"))
             {
              //We encountered something like |End Site|
              if (debug) {fprintf(stderr,"-E-");}
              if (InputParser_WordCompareAuto(ipcB,1,"Site"))
                   {
                    //---------------------------------------------------------------------------------------------------------------------
                    //------------------------------------------------- END SITE ----------------------------------------------------
                    //---------------------------------------------------------------------------------------------------------------------
                    if (debug) {fprintf(stderr,"-S-");}
                    if (jNum>0)
                    {
                      char parentJointName[MAX_BVH_JOINT_NAME+1];
                      snprintf(parentJointName,sizeof(parentJointName),"%s",bvhMotion->jointHierarchy[jNum-1].jointName);
                      int ret = snprintf(bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME,"EndSite_%s",parentJointName);

                      if (ret < 0)
                       {
                         fprintf(stderr,RED "A huge joint name was encountered please note that this might mean the joint names are not correct..\n" NORMAL );
                       }
                    } else
                    {
                      snprintf(bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME,"EndSite");
                    }

                    if (jNum<bvhMotion->MAX_jointHierarchySize)
                    {
                     bvhMotion->jointHierarchy[jNum].isEndSite=1;
                     bvhMotion->jointHierarchy[jNum].hierarchyLevel = hierarchyLevel;
                     //Update lookup table to remember ordering
                     bvhMotion->jointToMotionLookup[jNum].jointMotionOffset  = bvhMotion->numberOfValuesPerFrame;
                     currentJoint=jNum;
                     ++jNum;
                    } else
                    {
                     errorReachedMemoryLimit(jNum);
                    }
                   //---------------------------------------------------------------------------------------------------------------------
                   //---------------------------------------------------------------------------------------------------------------------
                   //---------------------------------------------------------------------------------------------------------------------
                   }
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"CHANNELS"))
             {
                  //---------------------------------------------------------------------------------------------------------------------
                  //------------------------------------------------- CHANNELS ----------------------------------------------------------
                  //---------------------------------------------------------------------------------------------------------------------
                  //Reached something like  |CHANNELS 3 Zrotation Xrotation Yrotation| declaration
                  if (debug) {fprintf(stderr,"-C");}

                  //Remember the parentID as shorthand..
                  unsigned int parentID=bvhMotion->jointHierarchy[currentJoint].parentJoint;

                  //Read number of Channels
                  unsigned int parsedChannels = numberOfTokensInBVHFileLineB-2;
                  unsigned int loadedChannels = InputParser_GetWordInt(ipcB,1);
                  bvhMotion->jointHierarchy[currentJoint].loadedChannels=loadedChannels;
                  if (debug) {fprintf(stderr,"(%u)-",loadedChannels);}

                  if ( parsedChannels != loadedChannels)
                  {
                      fprintf(stderr,RED "Encountered %u channels while %u channels declared in file `%s`\n" NORMAL,parsedChannels,loadedChannels,line);
                      fprintf(stderr,RED "Sticking with declared channel number \n" NORMAL);
                  }

                  //In any case wipe channels to make sure they are clean
                  memset(bvhMotion->jointHierarchy[currentJoint].channelType,0,sizeof(char) * BVH_VALID_CHANNEL_NAMES);

                  if (loadedChannels<BVH_VALID_CHANNEL_NAMES)
                  {
                     if (debug) {fprintf(stderr,"\nJOINT %u (%s) : ",currentJoint,bvhMotion->jointHierarchy[currentJoint].jointName);}
                     //Now to store the channel labels
                     unsigned int cL=0; //Channel To Load

                     //Add resolve
                     //----------------------------------------------------
                     for (cL=0; cL<BVH_VALID_CHANNEL_NAMES; cL++)
                     {
                            bvhMotion->jointHierarchy[currentJoint].resolveChannelType[cL]=BVH_INVALID_CHANNEL;
                     }
                     //----------------------------------------------------
                     for (cL=0; cL<loadedChannels; cL++)
                       {
                         //For each declared channel we need to enumerate the label to a value
                         unsigned int thisChannelID = enumerateInputParserChannel(ipcB,2+cL);
                         bvhMotion->jointHierarchy[currentJoint].channelType[cL]=thisChannelID;
                         bvhMotion->jointHierarchy[currentJoint].resolveChannelType[thisChannelID]=cL;
                       }
                     //----------------------------------------------------

                     //Derive the rotation order..
                     bvhMotion->jointHierarchy[currentJoint].channelRotationOrder = enumerateChannelOrder(bvhMotion,currentJoint);


                     //Fast (and slightly ugly) hack to transport angles like regular euler angles..
                     if (bvhMotion->jointHierarchy[currentJoint].hasRodriguesRotation)
                     {
                       for (cL=0; cL<loadedChannels; cL++)
                       {
                         //For each declared channel we need to enumerate the label to a value
                         if ( bvhMotion->jointHierarchy[currentJoint].channelType[cL]==BVH_RODRIGUES_X)
                              {  bvhMotion->jointHierarchy[currentJoint].channelType[cL]=BVH_ROTATION_X; }

                         if ( bvhMotion->jointHierarchy[currentJoint].channelType[cL]==BVH_RODRIGUES_Y)
                              {  bvhMotion->jointHierarchy[currentJoint].channelType[cL]=BVH_ROTATION_Y; }

                         if ( bvhMotion->jointHierarchy[currentJoint].channelType[cL]==BVH_RODRIGUES_Z)
                              {  bvhMotion->jointHierarchy[currentJoint].channelType[cL]=BVH_ROTATION_Z; }
                       }
                       //--------------------------------------------------------------------------
                     }

                     //We can now update lookup tables..
                     for (cL=0; cL<loadedChannels; cL++)
                       {
                         //For each declared channel we need to enumerate the label to a value
                         unsigned int thisChannelID = bvhMotion->jointHierarchy[currentJoint].channelType[cL];

                         if (debug) {fprintf(stderr,"#%u %s=%u ",cL,channelNames[thisChannelID],bvhMotion->numberOfValuesPerFrame);}

                         //Update jointToMotion Lookup Table..
                         bvhMotion->jointToMotionLookup[currentJoint].channelIDMotionOffset[thisChannelID] = bvhMotion->numberOfValuesPerFrame;

                         //Update motionToJoint Lookup Table..
                         bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].channelID = thisChannelID;
                         bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].jointID   = currentJoint;
                         bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].parentID  = parentID;

                         ++bvhMotion->numberOfValuesPerFrame;
                       }

                     if (debug) {fprintf(stderr,"\n");}


                     //Done
                  }//The number of channels is small enough to be handled
                   else
                  {
                      fprintf(stderr,RED "Specified joint has too many channels (has %u, max is %d) and cannot be handled!\n" NORMAL,loadedChannels,BVH_VALID_CHANNEL_NAMES);
                  }
                  //---------------------------------------------------------------------------------------------------------------------
                  //---------------------------------------------------------------------------------------------------------------------
                  //---------------------------------------------------------------------------------------------------------------------
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"OFFSET"))
             {
              //---------------------------------------------------------------------------------------------------------------------
              //------------------------------------------------- OFFSET ------------------------------------------------------------
              //---------------------------------------------------------------------------------------------------------------------
              //Reached something like |OFFSET 3.91 0.00 0.00|
              if (debug) {fprintf(stderr,"-O-");}

              //Store offsets..
              //TODO: could check numberOfTokensInBVHFileLineB to make sure all offsets are present..
              if (numberOfTokensInBVHFileLineB==4)
                {
                 bvhMotion->jointHierarchy[currentJoint].offset[0]=InputParser_GetWordFloat(ipcB,1);
                 bvhMotion->jointHierarchy[currentJoint].offset[1]=InputParser_GetWordFloat(ipcB,2);
                 bvhMotion->jointHierarchy[currentJoint].offset[2]=InputParser_GetWordFloat(ipcB,3);
                } else
                {
                 fprintf(stderr,RED "Incorrect number of offset arguments..\n" NORMAL);
                 bvhMotion->jointHierarchy[currentJoint].offset[0]=0.0;
                 bvhMotion->jointHierarchy[currentJoint].offset[1]=0.0;
                 bvhMotion->jointHierarchy[currentJoint].offset[2]=0.0;
                }

              //Create 4x4 Matrix that contains the static transformation ( translation ) of the joint
              bvh_populateStaticTransformationOfJoint(bvhMotion,currentJoint);
              //---------------------------------------------------------------------------------------------------------------------
              //---------------------------------------------------------------------------------------------------------------------
              //---------------------------------------------------------------------------------------------------------------------
             } else
         if ( (InputParser_WordCompareAuto(ipcB,0,"{")) || (thisLineOnlyHasX(line,'{')) )
             {
              //We reached an { so we need to finish our joint OR root declaration
              if (debug) {fprintf(stderr,"-{%u-",hierarchyLevel);}
              if (
                  bvhMotion->jointHierarchy[currentJoint].hierarchyLevel == hierarchyLevel
                 )
                 {
                  bvhMotion->jointHierarchy[currentJoint].hasBrace=1;

                  if (
                      getParentJoint(
                                     bvhMotion,
                                     currentJoint,
                                     hierarchyLevel,
                                     &bvhMotion->jointHierarchy[currentJoint].parentJoint
                                    )
                      )
                   {
                     //We have a parent joint..!
                     if ( bvhMotion->jointHierarchy[currentJoint].isEndSite)
                      { //If current joint is an EndSite we must inform parent joint that it has an End Site
                       unsigned int parentID=bvhMotion->jointHierarchy[currentJoint].parentJoint;
                       bvhMotion->jointHierarchy[parentID].hasEndSite=1;
                      }
                   } else
                   {
                    fprintf(stderr,RED "Bug: could not find a parent joint..\n" NORMAL);
                   }
                 } else
                 {
                  fprintf(stderr,RED "Bug: HierarchyLevel not set at braces..\n" NORMAL );
                 }
               ++hierarchyLevel;
              } else
         if ( (InputParser_WordCompareAuto(ipcB,0,"}")) || (thisLineOnlyHasX(line,'}') ) )
             {
              //We reached an } so we pop one hierarchyLevel
              if (hierarchyLevel>0)
                 {
                  --hierarchyLevel;
                 } else
                 {
                   fprintf(stderr,RED "Erroneous BVH hierarchy..\n" NORMAL);
                 }

              if (hierarchyLevel==0)
              {
                //We are done..
                done=1;
              }

              //-------------------------------------
              if (debug) {fprintf(stderr,"-%u}-",hierarchyLevel);}
             }
          else
         {
            //Unexpected input..
            fprintf(stderr,"BVH Header, Unexpected line num (%u) of length %zd :\n" , bvhMotion->linesParsed , read);
            fprintf(stderr,"%s\n", line);
            //exit(0);
         }

         } // We have header content
       } // We are at header section
      } //We have content
    } //We have line input from file

   //Free incoming line buffer..
   if (line) { free(line); line=0;}

   if (hierarchyLevel!=0)
   {
     fprintf(stderr,RED "Missing } braces..\n" NORMAL);
     atHeaderSection = 0;
   }

   bvhMotion->jointHierarchySize = jNum;


   if (bvhMotion->jointHierarchySize!=bvhMotion->MAX_jointHierarchySize)
   {
     fprintf(stderr,RED "BUG: We allocated more memory than what was needed for joint hierarchy..\n" NORMAL);
     fprintf(stderr,RED "used %u vs allocated %u ..\n" NORMAL,bvhMotion->jointHierarchySize,bvhMotion->MAX_jointHierarchySize);
     exit(0);
   }

   if (bvhMotion->motionValuesSize != bvhMotion->numberOfValuesPerFrame)
   {
     fprintf(stderr,RED "BUG: We allocated more memory than what was needed for values hierarchy..\n" NORMAL);
     fprintf(stderr,RED "used %u vs allocated %u ..\n" NORMAL,bvhMotion->numberOfValuesPerFrame,bvhMotion->motionValuesSize);
     exit(0);
   }

   //Pretend Motion Values Size is 0 numberOfValuesPerFrame will hold the value now
   //It will be set @ readBVHMotion call..
   bvhMotion->motionValuesSize=0;

   InputParser_Destroy(ipc);
   InputParser_Destroy(ipcB);
   //We need file to be open..!
  }


  fprintf(
           stderr,
           "\nNumber of Values Per Frame: %u\n",
           bvhMotion->numberOfValuesPerFrame
          );

 return atHeaderSection;
}





int readBVHMotion(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{
  int atMotionSection=0;

  if (fd!=0)
  {
   struct InputParserC * ipc = InputParser_Create(MAX_BVH_FILE_LINE_SIZE,4);

   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,10);
   InputParser_SetDelimeter(ipc,2,13);
   InputParser_SetDelimeter(ipc,3,0);

    char str[MAX_BVH_FILE_LINE_SIZE+1]={0};
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    while ((read = getline(&line, &len, fd)) != -1)
    {
       ++bvhMotion->linesParsed;

       //fprintf(stderr,"Retrieved line of length %zu :\n %s", read,line);

       int num = InputParser_SeperateWords(ipc,line,1);

       //fprintf(stderr,"Has %u arguments\n",num);
       //InputParser_GetWord(ipc,0,str,512);
       //fprintf(stderr,"Word0=`%s`",str);
       //InputParser_GetWord(ipc,1,str,512);
       //fprintf(stderr,"Word1=`%s`",str);
       //InputParser_GetWord(ipc,2,str,512);
       //fprintf(stderr,"Word2=`%s`",str);

      if (num>0)
      { //We have content..
       if (!atMotionSection)
       {
          if (InputParser_WordCompareAuto(ipc,0,"MOTION"))
               {
                //fprintf(stderr,"Found Motion Section (%s)..\n",line);
                atMotionSection=1;
               }
       } else
       {
         if (InputParser_WordCompareAuto(ipc,0,"Frames"))
              {
                //fprintf(stderr,"Frames (%s)..\n",line);
                bvhMotion->numberOfFrames = InputParser_GetWordInt(ipc,1);
                //fprintf(stderr,"Frames number (%u)..\n",bvhMotion->numberOfFrames);

                //Take care of multiple uses of the same structure..
                if (bvhMotion->motionValues!=0)
                 {
                  fprintf(stderr,"Motion values allocation was dirty when we reached the new Frames declaration\n");
                  //BUG: If a Frames structure is not hit the motion values allocation will not be updated..
                  bvhMotion->motionValuesSize = 0;
                  free(bvhMotion->motionValues);
                  bvhMotion->motionValues = 0;
                 }
              } else
         if (InputParser_WordCompareAuto(ipc,0,"Frame Time"))
             {
                bvhMotion->frameTime = InputParser_GetWordFloat(ipc,1);
             }  else
         {
           if (bvhMotion->motionValues==0)
           {
             //If we haven't yet allocated a motionValues array we need to do so now..!
             bvhMotion->motionValuesSize = bvhMotion->numberOfFrames * bvhMotion->numberOfValuesPerFrame;
             fprintf(stderr,"Allocating %lu bytes of memory to hold BVH motions \n",(unsigned long) sizeof(float) * bvhMotion->motionValuesSize);
             bvhMotion->motionValues = (float*)  malloc(sizeof(float) * (1+bvhMotion->motionValuesSize));
             if (bvhMotion->motionValues==0)
             {
               fprintf(stderr,"Failed to allocate enough memory for motion values..\n");
             }
           }

           //This is motion input
           InputParser_GetWord(ipc,0,str,MAX_BVH_FILE_LINE_SIZE);
           pushNewBVHMotionState(bvhMotion,str);
           str[0]=0;//Clean up str
         }
       }
       }
    }

   //Free incoming line buffer..
   if (line) { free(line); line=0; }

   InputParser_Destroy(ipc);
  }

  fprintf(
           stderr,
           "Frames: %u(%u) / Frame Time : %0.4f\n",
           bvhMotion->numberOfFrames,
           bvhMotion->numberOfFramesEncountered,
           bvhMotion->frameTime
          );

  return (atMotionSection);
}

