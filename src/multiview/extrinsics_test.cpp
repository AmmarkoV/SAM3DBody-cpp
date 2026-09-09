// extrinsics_test.cpp — synthetic 3-camera recovery check for solve_extrinsics.
//   cam0 = reference (world).  cam0&cam1 share marker 10; cam1&cam2 share marker
//   11; cam0&cam2 share nothing -> cam2 must be placed via the cam1 chain.
//   We synthesise each camera's view of the markers from known ground-truth
//   world poses and check the solver recovers the camera world poses.
// Exits 0 on success, 1 on failure.

#include "extrinsics.h"

#include <cmath>
#include <cstdio>

using mv::Mat4;

static Mat4 mul(const Mat4& A, const Mat4& B)
{
    Mat4 C{};
    for (int r=0;r<4;r++) for (int c=0;c<4;c++)
    { double s=0; for(int k=0;k<4;k++) s+=A[r*4+k]*B[k*4+c]; C[r*4+c]=s; }
    return C;
}
static Mat4 inv_rigid(const Mat4& T)
{
    double R[9]={T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10]};
    double t[3]={T[3],T[7],T[11]};
    double Rt[9]={R[0],R[3],R[6], R[1],R[4],R[7], R[2],R[5],R[8]};
    double ti[3]={ -(Rt[0]*t[0]+Rt[1]*t[1]+Rt[2]*t[2]),
                   -(Rt[3]*t[0]+Rt[4]*t[1]+Rt[5]*t[2]),
                   -(Rt[6]*t[0]+Rt[7]*t[1]+Rt[8]*t[2]) };
    return Mat4{ Rt[0],Rt[1],Rt[2],ti[0], Rt[3],Rt[4],Rt[5],ti[1], Rt[6],Rt[7],Rt[8],ti[2], 0,0,0,1 };
}
static Mat4 T_rotY(double a,double x,double y,double z)
{ double c=cos(a),s=sin(a); return Mat4{ c,0,s,x, 0,1,0,y, -s,0,c,z, 0,0,0,1 }; }
static Mat4 T_rotZ(double a,double x,double y,double z)
{ double c=cos(a),s=sin(a); return Mat4{ c,-s,0,x, s,c,0,y, 0,0,1,z, 0,0,0,1 }; }
static Mat4 T_rotX(double a,double x,double y,double z)
{ double c=cos(a),s=sin(a); return Mat4{ 1,0,0,x, 0,c,-s,y, 0,s,c,z, 0,0,0,1 }; }

static double maxdiff(const Mat4& A, const Mat4& B)
{ double m=0; for(int i=0;i<16;i++) m=std::max(m, std::fabs(A[i]-B[i])); return m; }

int main()
{
    const double D2R = M_PI/180.0;
    // Ground-truth camera world poses (T_world<-cam):
    Mat4 W0 = T_rotY(0,0,0,0);                 // reference
    Mat4 W1 = T_rotY( 20*D2R, 1.0, 0.0, 0.2);
    Mat4 W2 = T_rotY(-35*D2R, -0.5, 0.1, 1.0);
    // Marker world poses (T_world<-marker):
    Mat4 MA = T_rotX( 15*D2R, 0.30, 0.00, 0.50);   // id 10
    Mat4 MB = T_rotZ(-25*D2R,-0.20, 0.05, 0.80);   // id 11

    // Each camera sees a marker as T_cam<-marker = inv(W_cam) * (W_marker).
    mv::StaticPoseAccumulator a0, a1, a2;
    a0.add_pose(10, mul(inv_rigid(W0), MA));
    a1.add_pose(10, mul(inv_rigid(W1), MA));
    a1.add_pose(11, mul(inv_rigid(W1), MB));
    a2.add_pose(11, mul(inv_rigid(W2), MB));

    std::vector<mv::CameraStaticPoses> cams(3);
    cams[0].pose = a0.poses();
    cams[1].pose = a1.poses();
    cams[2].pose = a2.poses();

    std::string report;
    auto ext = mv::solve_extrinsics(cams, /*reference=*/0, &report);
    fprintf(stderr, "%s", report.c_str());

    int fail = 0;
    auto chk = [&](int i, const Mat4& gt){
        if (!ext[i].placed) { fprintf(stderr,"  FAIL cam %d not placed\n", i); fail=1; return; }
        double d = maxdiff(ext[i].T_world_cam, gt);
        fprintf(stderr, "  cam %d max|ΔT|=%.2e %s\n", i, d, d<1e-9?"ok":"FAIL");
        if (d >= 1e-9) fail = 1;
    };
    chk(0, W0);
    chk(1, W1);     // via shared marker 10 with the reference
    chk(2, W2);     // via the cam1 chain (marker 11), no direct share with cam0

    fprintf(stderr, fail ? "\nextrinsics_test: FAILED\n" : "\nextrinsics_test: OK\n");
    return fail;
}
