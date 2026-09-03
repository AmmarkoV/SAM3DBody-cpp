/** @file model_loader_transform_joints.h
 *  @brief  TRIModels loader/writer and basic functions
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer

            The purpose of this code is to perform all the necessary transformations on TRI Models based on their bones and poses we want to achieve
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
#define MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED


#include "model_loader_tri.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ── MHR Linear Blend Skinning ─────────────────────────────────────────────── */

struct MHR_LBS_Data {
    int n_joints;       /* 127  */
    int n_skin;         /* 51337 */
    int n_verts;        /* 18439 */
    int n_shape_pc;     /* 45   */
    int n_face_pc;      /* 72   */
    int pt_rows;        /* 889 = n_joints * 7 */
    int pt_cols;        /* 249  */
    int n_scale_pc;     /* 28   (0 if not loaded) */
    int n_scale_out;    /* 68   (0 if not loaded) */
    int n_hand_pca;     /* 54   (0 if not loaded; v3+) */
    int n_hand_out;     /* 27   per-hand model_params dim (v3+) */

    float *PT;                  /* [pt_rows × pt_cols]          */
    float *joint_offsets;       /* [n_joints × 3]               */
    float *joint_prerotations;  /* [n_joints × 4]  XYZW         */
    int   *joint_parents;       /* [n_joints]  -1 = root        */
    float *inv_bind_pose;       /* [n_joints × 8]  tx,ty,tz,qx,qy,qz,qw,scale */
    int   *skin_joint_idx;      /* [n_skin]                     */
    float *skin_weights;        /* [n_skin]                     */
    int   *skin_vert_idx;       /* [n_skin]                     */
    float *base_shape;          /* [n_verts × 3]                */
    float *shape_vectors;       /* [n_shape_pc × n_verts × 3]  */
    float *face_vectors;        /* [n_face_pc  × n_verts × 3]  */
    float *scale_mean;          /* [n_scale_out]  NULL if not loaded */
    float *scale_comps;         /* [n_scale_pc × n_scale_out]  NULL if not loaded */
    /* v3: hand pose PCA + per-hand joint index tables */
    float *hand_pose_mean;      /* [n_hand_pca]                  NULL if not loaded */
    float *hand_pose_comps;     /* [n_hand_pca × n_hand_pca]     NULL if not loaded */
    int   *hand_joint_idxs_left;/* [n_hand_out]                  NULL if not loaded */
    int   *hand_joint_idxs_right;/*[n_hand_out]                  NULL if not loaded */

    /* Pose correctives: sparse 2-layer network applied to unposed mesh before LBS.
     * Loaded from correctives.bin via mhr_correctives_load().
     * NULL when not loaded. */
    int    corr_n_feat;         /* (n_joints-2)*6 = 750          */
    int    corr_n_hidden;       /* 3000                          */
    int    corr_n_out;          /* n_verts*3 = 55317             */
    int    corr_nnz1;           /* nnz in sparse layer 1         */
    int    corr_nnz2;           /* nnz in sparse layer 2 (dense but pruned) */
    int   *corr_sp1_row;        /* [corr_nnz1]  row idx, layer 1 */
    int   *corr_sp1_col;        /* [corr_nnz1]  col idx, layer 1 */
    float *corr_sp1_val;        /* [corr_nnz1]  values,  layer 1 */
    int   *corr_sp2_row;        /* [corr_nnz2]  row idx, layer 2 */
    int   *corr_sp2_col;        /* [corr_nnz2]  col idx, layer 2 */
    float *corr_sp2_val;        /* [corr_nnz2]  values,  layer 2 */
};

/* Load body_model.lbs produced by tools/extract_lbs_data.py.
 * Returns NULL on failure. Caller must call mhr_lbs_free(). */
struct MHR_LBS_Data *mhr_lbs_load(const char *path);

/* Load correctives.bin produced by tools/export_correctives.py and attach to d.
 * Safe to call on d returned by mhr_lbs_load(); skipped if file not found.
 * Returns 1 on success, 0 if file missing/incompatible (non-fatal). */
int mhr_correctives_load(struct MHR_LBS_Data *d, const char *path);

/* Free all buffers allocated by mhr_lbs_load(). */
void mhr_lbs_free(struct MHR_LBS_Data *d);

/* Run the full LBS forward pass for one person per frame.
 *   model_params  [204]        from MHRResult.mhr_model_params
 *   shape_coeffs  [n_shape_pc] from MHRResult.shape
 *   face_coeffs   [n_face_pc]  from MHRResult.face_params
 *   out_verts     [n_verts*3]  caller-allocated output (float[18439*3])
 *   out_joints    [n_joints*3] caller-allocated output (float[127*3]), may be NULL
 *   out_joint_quats [n_joints*4] caller-allocated output (float[127*4], XYZW),
 *                 may be NULL. Global joint ROTATION (g_q from the FK step),
 *                 in the model's own unflipped frame — matches Python
 *                 mhr_forward(..., return_joint_rotations=True)'s
 *                 joint_global_rots directly (no Y/Z flip applied, unlike
 *                 out_joints/out_verts). Needed for the refined-pose wrist-IK
 *                 fusion (see PLAN.md, issue #15) — the position-only output
 *                 doesn't carry enough information to solve for a joint's
 *                 local rotation given a desired global rotation.
 * Applies pose correctives from d if mhr_correctives_load() was called.
 * Returns 1 on success, 0 on failure. */
int mhr_lbs_compute(const struct MHR_LBS_Data *d,
                    const float *model_params,
                    const float *shape_coeffs,
                    const float *face_coeffs,
                    float       *out_verts,
                    float       *out_joints,
                    float       *out_joint_quats);

/* ── TRI bone transforms ────────────────────────────────────────────────────── */

/**
* @brief This is the maximum number of bones per vertice this is needed to allocate correctly the arrays on TRI_Bones_Per_Vertex_Vertice_Item , 4 is
         a logical value..
* @ingroup TRI
*/
#define MAX_BONES_PER_VERTICE 4
struct TRI_Bones_Per_Vertex_Vertice_Item
{
  unsigned int bonesOfthisVertex;
  float weightsOfThisVertex[MAX_BONES_PER_VERTICE];
  unsigned int indicesOfThisVertex[MAX_BONES_PER_VERTICE];
  unsigned int boneIDOfThisVertex[MAX_BONES_PER_VERTICE];
};


/**
* @brief A different way to store TRI bones , per vertex ( for shader pose configuration )
* @ingroup TRI
*/
struct TRI_Bones_Per_Vertex
{
  unsigned int numberOfBones;
  unsigned int numberOfVertices;
  unsigned int maxBonesPerVertex;

  struct TRI_Bones_Per_Vertex_Vertice_Item * bonesPerVertex;
};



struct TRI_Bones_Per_Vertex * allocTransformTRIBonesToVertexBoneFormat(struct TRI_Model * in);
void freeTransformTRIBonesToVertexBoneFormat(struct TRI_Bones_Per_Vertex * in);



/**
* @brief Get the palette that colorCodeBones will use to reflect bone IDs
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
float * generatePalette(struct TRI_Model * in);





unsigned int * convertTRIBonesToParentList(struct TRI_Model * in , unsigned int * outputNumberOfBones);

/**
* @brief Populate in->bone[x].info->x/y/z with the centers of the bone based on the current setup of the
         model..
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
* @param  output number of bones allocated ..!
* @retval 0=Failure or else a pointer to an array of float triplets with x/y/z locations of each bone.
* @bug  Please note that prior calling this function the program should have done a applyVertexTransformation or doModelTransform to set up the vertices , because this function just calculates
the average of each bone. Please note that the output is in the coordinate space of the binding pose model and needs to be transformed/projected etc
according to the real location of the mesh , this function is also quite resource heavy and needs to be improved..
*/
float * convertTRIBonesToJointPositions(struct TRI_Model * in , unsigned int * outputNumberOfJoints);


unsigned int  * getClosestVertexToJointPosition(struct TRI_Model * in , float * joints , unsigned int numberOfJoints);


int tri_colorCodeTexture(struct TRI_Model * in, unsigned int x, unsigned int y, unsigned int width,unsigned int height);



int setTRIJointRotationOrder(
                              struct TRI_Model * in ,
                              unsigned int jointToChange ,
                              unsigned int rotationOrder
                             );

int getTRIJointRotationOrder(
                             struct TRI_Model * in ,
                             unsigned int jointToChange ,
                             unsigned int rotationOrder
                            );




/**
* @brief Populate in->bone[x].info->x/y/z with the centers of the bone based on the current setup of the
         model..
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
int setTRIModelBoneInitialPosition(struct TRI_Model * in);



/**
* @brief Alter color information of model to reflect bone IDs
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
void colorCodeBones(struct TRI_Model * in);





/**
* @brief Transform a TRI Joint using just 3 Euler Angles using ZYX order..!
* @ingroup TRI
* @param  input TRI structure with the loaded model
* @param  allocated matrix array that will be altered
* @param  size of allocated matrix array

* @param  The joint to select in->bones[jointToChange].boneName to see what it was
* @param  Rotation in Euler Angle axis X
* @param  Rotation in Euler Angle axis Y
* @param  Rotation in Euler Angle axis Z
*/
void transformTRIJoint(
                        struct TRI_Model * in ,
                        float * jointData ,
                        unsigned int jointDataSize ,

                        unsigned int jointToChange ,
                        float rotEulerX ,
                        float rotEulerY ,
                        float rotEulerZ
                      );



/**
* @brief Allocate all the 4x4 matrices needed to control a TRI_Model , this call also initializes them so they are ready for use..! , they need to be freed after being used
* @ingroup TRI
* @param  input TRI structure with the loaded model we want to allocate matrices for
* @param  output number of 4x4 matrices allocated
* @retval 0=Failure or else a pointer to an array of 4x4 matrices
*/
float * mallocModelTransformJoints(
                                    struct TRI_Model * triModelInput ,
                                    unsigned int * jointDataSizeOutput
                                  );



float * mallocModelTransformJointsEulerAnglesDegrees(
                                                      struct TRI_Model * triModelInput ,
                                                      float * jointData ,
                                                      unsigned int jointDataSize ,
                                                      unsigned int method
                                                     );


int applyVertexTransformation( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn );

void printModelTransform(struct TRI_Model * in);

/**
* @brief  Do model transform based on joints
* @ingroup TRI
* @param  output TRI structure transformed as requested
* @param  input TRI structure
* @param  array of joints allocated using mallocModelTransformJoints
* @param  number of joints got from the jointDataSizeOutput of mallocModelTransformJoints
* @param  autodetect non-identity matrices on jointdata array and skip some calculations
* @param  use direct setting of final matrices ( only needed if you really know what you want in the final matrices )
* @param  perform vertex transforms ( if not triModelOut will be the same as triModelIn )
* @param  jointAxisConvention , 0 = default
* @retval 0=Failure,1=Success
*/
int doModelTransform(
                      struct TRI_Model * triModelOut ,
                      struct TRI_Model * triModelIn ,
                      float * joint4x4Data ,
                      unsigned int joint4x4DataSize ,
                      unsigned int autodetectAlteredMatrices,
                      unsigned int directSettingOfMatrices ,
                      unsigned int performVertexTransform  ,
                      unsigned int jointAxisConvention
                    );


#ifdef __cplusplus
}
#endif

#endif // MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
