// ════════════════════════════════════════════════════════════════════════════
//  extrinsics.h — camera extrinsics from shared STATIC ArUco markers
//
//  Each camera observes one or more static markers (the moving id<10 object
//  markers must be excluded by the caller).  A marker's pose in a camera frame
//  is T_cam←marker (from estimatePoseSingleMarkers: X_cam = R·X_marker + t).
//  Because the markers are static, that pose is time-invariant, so we average
//  many observations per (camera,marker) into one stable estimate — no temporal
//  sync needed.
//
//  Two cameras sharing a static marker M give the relative pose
//      T_a←b = T_a←M · (T_b←M)⁻¹.
//  Picking a reference camera as the world frame and walking the camera
//  co-visibility graph (BFS) yields T_world←cam for every connected camera.
//
//  Pure C++ (no OpenCV): SE(3) is a row-major 4×4 double array.  See
//  MULTIVIEW_PLAN.md.
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_EXTRINSICS_H_INCLUDED
#define MULTIVIEW_EXTRINSICS_H_INCLUDED

#include <array>
#include <map>
#include <string>
#include <vector>

namespace mv
{

using Mat4 = std::array<double,16>;   // row-major 4×4 homogeneous transform

// Accumulate repeated observations of static markers in ONE camera and average
// each (rotation via sign-aligned quaternion mean, translation via mean).
class StaticPoseAccumulator
{
public:
    // Add one observation of marker `id`: Rodrigues rotation + translation
    // (the rvec/tvec from estimatePoseSingleMarkers — T_cam←marker).
    void add(int id, const double rvec[3], const double tvec[3]);
    // Add one observation given the full T_cam←marker directly.
    void add_pose(int id, const Mat4& T_cam_marker);

    std::map<int, Mat4> poses()  const;   // id -> averaged T_cam←marker
    std::map<int, int>  counts() const { return count_; }

private:
    void accumulate(int id, const double R[9], const double t[3]);
    std::map<int,int>                   count_;
    std::map<int,std::array<double,3>>  tsum_;
    std::map<int,std::array<double,4>>  qsum_;   // sign-aligned quaternion sum (xyzw)
};

// One camera's averaged static-marker poses.
struct CameraStaticPoses
{
    std::map<int, Mat4> pose;    // marker id -> T_cam←marker
    std::map<int, int>  count;   // marker id -> #observations averaged
};

struct CamExtrinsic
{
    bool  placed = false;
    Mat4  T_world_cam{};         // world == the reference camera's frame
    int   hops = -1;             // BFS distance from the reference camera
};

// Solve T_world←cam for each input camera (world == camera `reference`).
// Cameras with no marker path to the reference come back placed=false.
// If `report` is non-null it receives a human-readable summary (edges, shared
// markers, and a marker-position cross-camera agreement residual in metres).
std::vector<CamExtrinsic> solve_extrinsics(const std::vector<CameraStaticPoses>& cams,
                                           int reference,
                                           std::string* report = nullptr);

// Small accessors on a 4×4 (used by the fusion front-end).
std::array<double,4> mat4_rotation_quat(const Mat4& T);                       // xyzw
std::array<double,3> mat4_transform_point(const Mat4& T, const std::array<double,3>& p);

} // namespace mv

#endif // MULTIVIEW_EXTRINSICS_H_INCLUDED
