// ════════════════════════════════════════════════════════════════════════════
//  calib.h — Stereolabs/Matlab `.calib` reader/writer (intrinsics + extrinsics)
//
//  A freshened, trimmed port of AmmarkoV's RGBDAcquisition calibration loader
//  (tools/Calibration/calibration.c).  The original also carried point-
//  projection / depth-registration helpers that pulled in RGBDAcquisition's
//  transform.h / undistort.h / AmMatrix — those are dropped here because the
//  multi-view pipeline does its geometry with OpenCV.  What remains is the
//  self-contained, dependency-free `.calib` parse/serialise + the struct.
//
//  The `.calib` format is line based, category-tagged (`%I` intrinsics,
//  `%D` distortion, `%T` translation, `%R` Rodrigues rotation, `%RT4*4` 4x4
//  extrinsic, `%NF` near/far, `%UNIT` depth unit, `%Width`, `%Height`); values
//  follow one per line.  Used by tracker.py and by the multi-view sync step
//  (see MULTIVIEW_PLAN.md).
//
//  @author Ammar Qammaz (AmmarkoV)
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_CALIB_H_INCLUDED
#define MULTIVIEW_CALIB_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

// Shorthand indices into struct calibration::intrinsic[9] (row-major 3x3 K).
enum calibIntrinsics
{
  CALIB_INTR_FX = 0,
  CALIB_INTR_FY = 4,
  CALIB_INTR_CX = 2,
  CALIB_INTR_CY = 5
};

// Camera calibration: intrinsics, distortion, extrinsics and frame dimensions.
struct calibration
{
  /* CAMERA INTRINSIC PARAMETERS */
  char  intrinsicParametersSet;
  float intrinsic[9];                     // row-major 3x3 K
  float k1, k2, p1, p2, k3;               // OpenCV distortion order

  /* CAMERA EXTRINSIC PARAMETERS */
  char  extrinsicParametersSet;
  float extrinsicRotationRodriguez[3];
  float extrinsicTranslation[3];
  float extrinsic[16];                    // row-major 4x4 (when %RT4*4 present)

  /* CAMERA DIMENSIONS (WHEN RENDERING) */
  float nearPlane, farPlane;
  unsigned int width;
  unsigned int height;

  float depthUnit;

  /* CONFIGURATION (board metadata, only written/read for provenance) */
  int   imagesUsed;
  int   boardWidth;
  int   boardHeight;
  float squareSize;
};

// Locale-safe atof: parses "3.14" regardless of the host LC_NUMERIC (some
// locales write "3,14", which a plain atof would truncate to 3).  Kept because
// it is the historical reason this loader is robust across machines.
float internationalAtof(const char * str);

// Populate a calibration with sane defaults for the given frame size.
int NullCalibration(unsigned int width, unsigned int height, struct calibration * calib);

// Derive intrinsics from a focal length + pixel size (e.g. PrimeSense devices).
int FocalLengthAndPixelSizeToCalibration(double focalLength, double pixelSize,
                                         unsigned int width, unsigned int height,
                                         struct calibration * calib);

// Incrementally read `.calib` values into an already-initialised struct.
int RefreshCalibration(const char * filename, struct calibration * calib);

// NullCalibration(width,height) then RefreshCalibration(filename).
int ReadCalibration(const char * filename, unsigned int width, unsigned int height,
                    struct calibration * calib);

// Serialise a calibration back to the `.calib` format.
int WriteCalibration(const char * filename, struct calibration * calib);

// Human-readable dump to stderr.
int PrintCalibration(struct calibration * calib);

#ifdef __cplusplus
}
#endif

#endif // MULTIVIEW_CALIB_H_INCLUDED
