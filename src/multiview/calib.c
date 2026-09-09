// ════════════════════════════════════════════════════════════════════════════
//  calib.c — see calib.h.
//
//  Trimmed port of RGBDAcquisition tools/Calibration/calibration.c: the parser,
//  serialiser and defaults only — the point-projection / depth-registration
//  helpers (and their transform.h / undistort.h / AmMatrix dependencies) are
//  intentionally not carried over, the multi-view pipeline uses OpenCV for that.
// ════════════════════════════════════════════════════════════════════════════

#include "calib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>

#define DEBUG_PRINT_EACH_CALIBRATION_LINE_READ 0

#define DEFAULT_WIDTH  640
#define DEFAULT_HEIGHT 480
#define DEFAULT_FX     535.423666
#define DEFAULT_FY     533.484666

#define MAX_LINE_CALIBRATION 1024

#define RESPECT_LOCALES 0   // should always be 0 — see internationalAtof()

static unsigned char warnUSLocale = 0;

// Force EN_US numeric locale so commas-as-decimal-separators ("3,14") don't
// silently truncate to integers when parsing on non-US machines.
static int forceUSLocaleToKeepOurSanity(void)
{
   if (!warnUSLocale)
      {
        fprintf(stderr,"Reinforcing EN_US locale to force commas to dots (This warning appears only one time)\n");
        warnUSLocale=1;
      }

   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}


int NullCalibration(unsigned int width,unsigned int height, struct calibration * calib)
{
  if (calib==0) { fprintf(stderr,"NullCalibration cannot empty a non allocated calibration structure \n"); return 0;  }

  calib->width=width;
  calib->height=height;

  calib->intrinsicParametersSet=0;
  calib->extrinsicParametersSet=0;

  calib->nearPlane=1.0;
  calib->farPlane=1000.0;

  calib->intrinsic[0]=0.0;  calib->intrinsic[1]=0.0;  calib->intrinsic[2]=0.0;
  calib->intrinsic[3]=0.0;  calib->intrinsic[4]=0.0;  calib->intrinsic[5]=0.0;
  calib->intrinsic[6]=0.0;  calib->intrinsic[7]=0.0;  calib->intrinsic[8]=1.0;

  calib->k1=0.0;  calib->k2=0.0; calib->p1=0.0; calib->p2=0.0; calib->k3=0.0;

  calib->extrinsicRotationRodriguez[0]=0.0; calib->extrinsicRotationRodriguez[1]=0.0; calib->extrinsicRotationRodriguez[2]=0.0;
  calib->extrinsicTranslation[0]=0.0; calib->extrinsicTranslation[1]=0.0; calib->extrinsicTranslation[2]=0.0;

  /*cx*/calib->intrinsic[CALIB_INTR_CX]  = (double) width/2;
  /*cy*/calib->intrinsic[CALIB_INTR_CY]  = (double) height/2;

  //-This is a bad initial estimation i guess :P
  /*fx*/ calib->intrinsic[CALIB_INTR_FX] = (double) (DEFAULT_FX * width) / DEFAULT_WIDTH;
  /*fy*/ calib->intrinsic[CALIB_INTR_FY] = (double) (DEFAULT_FY * height)  / DEFAULT_HEIGHT;
  //--------------------------------------------

  calib->depthUnit=1000.0; //Default is meters to millimeters

  return 1;
}


int FocalLengthAndPixelSizeToCalibration(double focalLength , double pixelSize ,unsigned int width,unsigned int height ,  struct calibration * calib)
{
  NullCalibration(width,height,calib);

  fprintf(stderr,"FocalLengthAndPixelSizeToCalibration(focalLength=%0.2f,pixelSize=%0.2f,width=%u,height=%u) = ",focalLength,pixelSize,width,height);

  if (pixelSize!=0)
  {
   calib->intrinsic[CALIB_INTR_FX] = (double) focalLength/pixelSize;
   calib->intrinsic[CALIB_INTR_FY] = (double) focalLength/pixelSize;
   calib->intrinsicParametersSet=1;
   fprintf(stderr,"fx : %0.2f fy : %0.2f \n",calib->intrinsic[CALIB_INTR_FX],calib->intrinsic[CALIB_INTR_FY]);
   return 1;
  }

  fprintf(stderr,"FocalLengthAndPixelSizeToCalibration(with focalLength %f and pixelSize %f) cannot yield a valid calibration\n",focalLength , pixelSize);
  return 0;
}


float internationalAtof(const char * str)
{
  #if RESPECT_LOCALES
   return atof(str);
  #else
  forceUSLocaleToKeepOurSanity();
  // sscanf("%f") parses a dot decimal regardless of the comma-decimal locales
  // (e.g. fr_FR) that would make a plain atof() drop the fractional part.
   float retVal=0.0;
   sscanf(str,"%f",&retVal);
   return retVal;
  #endif // RESPECT_LOCALES
}


int RefreshCalibration(const char * filename,struct calibration * calib)
{
  if ((filename==0)||(calib==0)) { return 0; }
  forceUSLocaleToKeepOurSanity();

  FILE * fp = 0;
  fp = fopen(filename,"r");
  if (fp == 0 ) {  return 0; }

  char line[MAX_LINE_CALIBRATION]={0};
  unsigned int lineLength=0;

  unsigned int i=0;

  unsigned int category=0;
  unsigned int linesAtCurrentCategory=0;


  while ( fgets(line,MAX_LINE_CALIBRATION,fp)!=0 )
   {
     {
     lineLength = strlen ( line );
     if ( lineLength > 0 ) {
                                 if (line[lineLength-1]==10) { line[lineLength-1]=0; }
                                 if (line[lineLength-1]==13) { line[lineLength-1]=0; }
                           }
     if ( lineLength > 1 ) {
                                 if (line[lineLength-2]==10) { line[lineLength-2]=0; }
                                 if (line[lineLength-2]==13) { line[lineLength-2]=0; }
                           }


     if (line[0]=='%') { linesAtCurrentCategory=0; }
     if ( (line[0]=='%') && (line[1]=='I') && (line[2]==0) )                   { category=1;    } else
     if ( (line[0]=='%') && (line[1]=='D') && (line[2]==0) )                   { category=2;    } else
     if ( (line[0]=='%') && (line[1]=='T') && (line[2]==0) )                   { category=3;    } else
     if ( (line[0]=='%') && (line[1]=='R') && (line[2]==0) )                   { category=4;    } else
     if ( (line[0]=='%') && (line[1]=='N') && (line[2]=='F') && (line[3]==0) ) { category=5;    } else
     if ( (line[0]=='%') && (line[1]=='U') && (line[2]=='N') && (line[3]=='I')
                         && (line[4]=='T') && (line[5]==0) )                   { category=6;    } else
     if ( (line[0]=='%') && (line[1]=='R') && (line[2]=='T') && (line[3]=='4')
                         && (line[4]=='*') && (line[5]=='4') && (line[6]==0) ) { category=7;    } else
     if ( (line[0]=='%') && (line[1]=='W') && (line[2]=='i') && (line[3]=='d')
                         && (line[4]=='t') && (line[5]=='h') && (line[6]==0) ) { category=8;    } else
     if ( (line[0]=='%') && (line[1]=='H') && (line[2]=='e') && (line[3]=='i') && (line[4]=='g')
                         && (line[5]=='h') && (line[6]=='t') && (line[7]==0))  { category=9;    } else
        {
          #if DEBUG_PRINT_EACH_CALIBRATION_LINE_READ
           fprintf(stderr,"Line %u ( %s ) is category %u lines %u \n",i,line,category,linesAtCurrentCategory);
          #endif

          if (category==1)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->intrinsic[0] = (double) internationalAtof(line); break;
             case 2 :  calib->intrinsic[1] = (double) internationalAtof(line); break;
             case 3 :  calib->intrinsic[2] = (double) internationalAtof(line); break;
             case 4 :  calib->intrinsic[3] = (double) internationalAtof(line); break;
             case 5 :  calib->intrinsic[4] = (double) internationalAtof(line); break;
             case 6 :  calib->intrinsic[5] = (double) internationalAtof(line); break;
             case 7 :  calib->intrinsic[6] = (double) internationalAtof(line); break;
             case 8 :  calib->intrinsic[7] = (double) internationalAtof(line); break;
             case 9 :  calib->intrinsic[8] = (double) internationalAtof(line); break;
           };
          } else
          if (category==2)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->k1 = (double) internationalAtof(line); break;
             case 2 :  calib->k2 = (double) internationalAtof(line); break;
             case 3 :  calib->p1 = (double) internationalAtof(line); break;
             case 4 :  calib->p2 = (double) internationalAtof(line); break;
             case 5 :  calib->k3 = (double) internationalAtof(line); break;
           };
          } else
          if (category==3)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicTranslation[0] = (double) internationalAtof(line); break;
             case 2 :  calib->extrinsicTranslation[1] = (double) internationalAtof(line); break;
             case 3 :  calib->extrinsicTranslation[2] = (double) internationalAtof(line); break;
           };
          } else
          if (category==4)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicRotationRodriguez[0] = (double) internationalAtof(line); break;
             case 2 :  calib->extrinsicRotationRodriguez[1] = (double) internationalAtof(line); break;
             case 3 :  calib->extrinsicRotationRodriguez[2] = (double) internationalAtof(line); break;
           };
          }else
          if (category==5)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->nearPlane = (double) internationalAtof(line); break;
             case 2 :  calib->farPlane  = (double) internationalAtof(line); break;
           };
          } else
          if (category==6)
          {
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->depthUnit = (double) internationalAtof(line); break;
           };
          } else
          if (category==7)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsic[0]  = (double) internationalAtof(line); break;
             case 2 :  calib->extrinsic[1]  = (double) internationalAtof(line); break;
             case 3 :  calib->extrinsic[2]  = (double) internationalAtof(line); break;
             case 4 :  calib->extrinsic[3]  = (double) internationalAtof(line); break;
             case 5 :  calib->extrinsic[4]  = (double) internationalAtof(line); break;
             case 6 :  calib->extrinsic[5]  = (double) internationalAtof(line); break;
             case 7 :  calib->extrinsic[6]  = (double) internationalAtof(line); break;
             case 8 :  calib->extrinsic[7]  = (double) internationalAtof(line); break;
             case 9 :  calib->extrinsic[8]  = (double) internationalAtof(line); break;
             case 10:  calib->extrinsic[9]  = (double) internationalAtof(line); break;
             case 11:  calib->extrinsic[10] = (double) internationalAtof(line); break;
             case 12:  calib->extrinsic[11] = (double) internationalAtof(line); break;
             case 13:  calib->extrinsic[12] = (double) internationalAtof(line); break;
             case 14:  calib->extrinsic[13] = (double) internationalAtof(line); break;
             case 15:  calib->extrinsic[14] = (double) internationalAtof(line); break;
             case 16:  calib->extrinsic[15] = (double) internationalAtof(line); break;
           };
          } else
          if (category==8)
          {
             switch(linesAtCurrentCategory)
             {
              case 1: calib->width = (unsigned int) atoi(line); break;
             }
          } else
          if (category==9)
          {
            switch(linesAtCurrentCategory)
             {
              case 1: calib->height = (unsigned int) atoi(line); break;
             }
          }
        }

     ++linesAtCurrentCategory;
     ++i;
     line[0]=0;
     }
   }

  fclose(fp);

  return 1;
}


int ReadCalibration(const char * filename,unsigned int width,unsigned int height,struct calibration * calib)
{
  if ((filename==0)||(calib==0)) { return 0; }
  //First free
  NullCalibration(width,height,calib);
  return RefreshCalibration(filename,calib);
}


int PrintCalibration(struct calibration * calib)
{
  fprintf(stderr, "---------------------------------------------------------------------\n");
  if (calib==0) { fprintf(stderr,"No calibration structure provided for printout \n"); return 0; }
  fprintf(stderr, "Dimensions ( %u x %u ) \n",calib->width,calib->height);
  fprintf(stderr, "fx %0.5f fy %0.5f cx %0.5f cy %0.5f\n",calib->intrinsic[CALIB_INTR_FX],calib->intrinsic[CALIB_INTR_FY],
                                                          calib->intrinsic[CALIB_INTR_CX],calib->intrinsic[CALIB_INTR_CY]);
  fprintf(stderr, "k1 %0.5f k2 %0.5f p1 %0.5f p2 %0.5f k3 %0.5f\n",calib->k1,calib->k2,calib->p1,calib->p2,calib->k3);

  fprintf(stderr, "Tx %0.5f %0.5f %0.5f \n",calib->extrinsicTranslation[0],calib->extrinsicTranslation[1],calib->extrinsicTranslation[2]);
  fprintf(stderr, "Rodriguez %0.5f %0.5f %0.5f \n",calib->extrinsicRotationRodriguez[0],calib->extrinsicRotationRodriguez[1],calib->extrinsicRotationRodriguez[2]);
  fprintf(stderr, "---------------------------------------------------------------------\n");

  return 0;
}


int WriteCalibration(const char * filename,struct calibration * calib)
{
  if ((filename==0)||(calib==0)) { return 0; }
  forceUSLocaleToKeepOurSanity();

  FILE * fp = 0;
  fp = fopen(filename,"w");
  if (fp == 0 ) {  return 0; }

    fprintf( fp, "%%Calibration File\n");
    fprintf( fp, "%%CameraID=0\n");
    fprintf( fp, "%%CameraNo=0\n");

    time_t t;
    time( &t );
    struct tm *t2 = localtime( &t );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );
    fprintf( fp, "%%Date=%s\n",buf);


    fprintf( fp, "%%ImageWidth=%u\n",calib->width);
    fprintf( fp, "%%ImageHeight=%u\n",calib->height);
    fprintf( fp, "%%Description=After %u images , board is %ux%u , square size is %f , aspect ratio %0.2f\n",
                                                    calib->imagesUsed,
                                                    calib->boardWidth,
                                                    calib->boardHeight,
                                                    calib->squareSize,
                                                    (double) calib->width/calib->height);


    fprintf( fp, "%%Intrinsics I[1,1], I[1,2], I[1,3], I[2,1], I[2,2], I[2,3], I[3,1], I[3,2] I[3,3] 3x3\n");
    fprintf( fp, "%%I\n");
    fprintf( fp, "%f\n",calib->intrinsic[0]); fprintf( fp, "%f\n",calib->intrinsic[1]); fprintf( fp, "%f\n",calib->intrinsic[2]);
    fprintf( fp, "%f\n",calib->intrinsic[3]); fprintf( fp, "%f\n",calib->intrinsic[4]); fprintf( fp, "%f\n",calib->intrinsic[5]);
    fprintf( fp, "%f\n",calib->intrinsic[6]); fprintf( fp, "%f\n",calib->intrinsic[7]); fprintf( fp, "%f\n",calib->intrinsic[8]);



    fprintf( fp, "%%Distortion D[1], D[2], D[3], D[4] D[5] \n");
    fprintf( fp, "%%D\n%f\n%f\n%f\n%f\n%f\n",calib->k1,calib->k2,calib->p1,calib->p2,calib->k3);

    if( calib->extrinsicParametersSet )
    {
       fprintf( fp, "%%Translation T.X, T.Y, T.Z\n");
       fprintf( fp, "%%T\n");
       fprintf( fp, "%f\n%f\n%f\n",calib->extrinsicTranslation[0],calib->extrinsicTranslation[1],calib->extrinsicTranslation[2]);

       fprintf( fp, "%%Rotation Vector (Rodrigues) R.X, R.Y, R.Z \n");
       fprintf( fp, "%%R\n");
       fprintf( fp, "%f\n%f\n%f\n",calib->extrinsicRotationRodriguez[0],calib->extrinsicRotationRodriguez[1],calib->extrinsicRotationRodriguez[2]);
     }

 fclose(fp);

 return 1;
}
