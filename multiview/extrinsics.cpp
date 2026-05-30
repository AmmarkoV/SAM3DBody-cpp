// ════════════════════════════════════════════════════════════════════════════
//  extrinsics.cpp — see extrinsics.h.  Pure-C++ SE(3) / quaternion math.
// ════════════════════════════════════════════════════════════════════════════

#include "extrinsics.h"

#include <cmath>
#include <cstdio>
#include <queue>
#include <utility>

namespace mv
{
namespace
{

// ─── small linear-algebra helpers (row-major) ───────────────────────────────
void rodrigues_to_R(const double r[3], double R[9])
{
    double th = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (th < 1e-12) { R[0]=R[4]=R[8]=1; R[1]=R[2]=R[3]=R[5]=R[6]=R[7]=0; return; }
    double x=r[0]/th, y=r[1]/th, z=r[2]/th;
    double c=std::cos(th), s=std::sin(th), C=1-c;
    R[0]=c+x*x*C;   R[1]=x*y*C-z*s; R[2]=x*z*C+y*s;
    R[3]=y*x*C+z*s; R[4]=c+y*y*C;   R[5]=y*z*C-x*s;
    R[6]=z*x*C-y*s; R[7]=z*y*C+x*s; R[8]=c+z*z*C;
}

void R_to_quat(const double m[9], double q[4])   // xyzw, Shepperd
{
    double tr = m[0]+m[4]+m[8];
    if (tr > 0) { double s=std::sqrt(tr+1.0)*2; q[3]=0.25*s; q[0]=(m[7]-m[5])/s; q[1]=(m[2]-m[6])/s; q[2]=(m[3]-m[1])/s; }
    else if (m[0]>m[4] && m[0]>m[8]) { double s=std::sqrt(1+m[0]-m[4]-m[8])*2; q[3]=(m[7]-m[5])/s; q[0]=0.25*s; q[1]=(m[1]+m[3])/s; q[2]=(m[2]+m[6])/s; }
    else if (m[4]>m[8]) { double s=std::sqrt(1+m[4]-m[0]-m[8])*2; q[3]=(m[2]-m[6])/s; q[0]=(m[1]+m[3])/s; q[1]=0.25*s; q[2]=(m[5]+m[7])/s; }
    else { double s=std::sqrt(1+m[8]-m[0]-m[4])*2; q[3]=(m[3]-m[1])/s; q[0]=(m[2]+m[6])/s; q[1]=(m[5]+m[7])/s; q[2]=0.25*s; }
    double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]); if(n<1e-12)n=1;
    for (int i=0;i<4;i++) q[i]/=n;
}

void quat_to_R(const double q[4], double m[9])    // xyzw
{
    double x=q[0],y=q[1],z=q[2],w=q[3];
    m[0]=1-2*(y*y+z*z); m[1]=2*(x*y-w*z);   m[2]=2*(x*z+w*y);
    m[3]=2*(x*y+w*z);   m[4]=1-2*(x*x+z*z); m[5]=2*(y*z-w*x);
    m[6]=2*(x*z-w*y);   m[7]=2*(y*z+w*x);   m[8]=1-2*(x*x+y*y);
}

Mat4 mat4_from_Rt(const double R[9], const double t[3])
{
    return Mat4{ R[0],R[1],R[2],t[0],  R[3],R[4],R[5],t[1],  R[6],R[7],R[8],t[2],  0,0,0,1 };
}
Mat4 identity4() { return Mat4{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; }

Mat4 mat4_mul(const Mat4& A, const Mat4& B)
{
    Mat4 C{};
    for (int r=0;r<4;r++) for (int c=0;c<4;c++)
    { double s=0; for (int k=0;k<4;k++) s+=A[r*4+k]*B[k*4+c]; C[r*4+c]=s; }
    return C;
}

Mat4 mat4_inv_rigid(const Mat4& T)   // assumes [R t; 0 1] with R orthonormal
{
    double R[9]={T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10]};
    double t[3]={T[3],T[7],T[11]};
    double Rt[9]={R[0],R[3],R[6], R[1],R[4],R[7], R[2],R[5],R[8]};   // transpose
    double ti[3]={ -(Rt[0]*t[0]+Rt[1]*t[1]+Rt[2]*t[2]),
                   -(Rt[3]*t[0]+Rt[4]*t[1]+Rt[5]*t[2]),
                   -(Rt[6]*t[0]+Rt[7]*t[1]+Rt[8]*t[2]) };
    return mat4_from_Rt(Rt, ti);
}

void transform_point(const Mat4& T, const double p[3], double o[3])
{
    for (int i=0;i<3;i++) o[i]=T[i*4+0]*p[0]+T[i*4+1]*p[1]+T[i*4+2]*p[2]+T[i*4+3];
}

// Sign-aligned quaternion + translation averaging of several rigid transforms.
struct RigidAvg
{
    double qs[4]={0,0,0,0}, ts[3]={0,0,0}; int n=0;
    void add(const Mat4& T)
    {
        double R[9]={T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10]};
        double q[4]; R_to_quat(R,q);
        if (n==0) { for(int k=0;k<4;k++) qs[k]=q[k]; }
        else { double d=qs[0]*q[0]+qs[1]*q[1]+qs[2]*q[2]+qs[3]*q[3]; double s=d<0?-1:1; for(int k=0;k<4;k++) qs[k]+=s*q[k]; }
        ts[0]+=T[3]; ts[1]+=T[7]; ts[2]+=T[11]; ++n;
    }
    Mat4 result() const
    {
        double q[4]={qs[0],qs[1],qs[2],qs[3]};
        double nn=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]); if(nn<1e-12)nn=1;
        for(int k=0;k<4;k++) q[k]/=nn;
        double R[9]; quat_to_R(q,R);
        double t[3]={ts[0]/n, ts[1]/n, ts[2]/n};
        return mat4_from_Rt(R,t);
    }
};

} // anon namespace

// ─── StaticPoseAccumulator ───────────────────────────────────────────────────
void StaticPoseAccumulator::accumulate(int id, const double R[9], const double t[3])
{
    double q[4]; R_to_quat(R,q);
    auto& qs = qsum_[id]; auto& ts = tsum_[id];
    if (count_[id]==0) { qs={q[0],q[1],q[2],q[3]}; ts={t[0],t[1],t[2]}; }
    else
    {
        double d=qs[0]*q[0]+qs[1]*q[1]+qs[2]*q[2]+qs[3]*q[3]; double s=d<0?-1:1;
        for(int k=0;k<4;k++) qs[k]+=s*q[k];
        for(int k=0;k<3;k++) ts[k]+=t[k];
    }
    count_[id]++;
}

void StaticPoseAccumulator::add(int id, const double rvec[3], const double tvec[3])
{
    double R[9]; rodrigues_to_R(rvec,R);
    accumulate(id, R, tvec);
}

void StaticPoseAccumulator::add_pose(int id, const Mat4& T)
{
    double R[9]={T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10]};
    double t[3]={T[3],T[7],T[11]};
    accumulate(id, R, t);
}

std::map<int, Mat4> StaticPoseAccumulator::poses() const
{
    std::map<int,Mat4> out;
    for (auto& kv : count_)
    {
        int id=kv.first, c=kv.second;
        auto q = qsum_.at(id);
        double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]); if(n<1e-12)n=1;
        for(double& x:q) x/=n;
        double R[9]; quat_to_R(q.data(),R);
        auto t=tsum_.at(id); double tt[3]={t[0]/c, t[1]/c, t[2]/c};
        out[id]=mat4_from_Rt(R,tt);
    }
    return out;
}

// ─── solver ──────────────────────────────────────────────────────────────────
std::vector<CamExtrinsic> solve_extrinsics(const std::vector<CameraStaticPoses>& cams,
                                           int reference, std::string* report)
{
    const int N=(int)cams.size();
    std::vector<CamExtrinsic> out(N);
    if (N==0 || reference<0 || reference>=N) return out;

    // Relative transforms T_a←b averaged over the markers a and b share.
    std::vector<std::vector<int>>             adj(N);
    std::map<std::pair<int,int>, Mat4>        rel;
    std::map<std::pair<int,int>, std::vector<int>> shared_of;
    for (int a=0;a<N;a++)
        for (int b=0;b<N;b++) if (a!=b)
        {
            std::vector<int> shared;
            for (auto& kv : cams[a].pose) if (cams[b].pose.count(kv.first)) shared.push_back(kv.first);
            if (shared.empty()) continue;
            RigidAvg avg;
            for (int m : shared)
                avg.add(mat4_mul(cams[a].pose.at(m), mat4_inv_rigid(cams[b].pose.at(m))));
            rel[{a,b}] = avg.result();
            shared_of[{a,b}] = shared;
            adj[a].push_back(b);
        }

    // BFS from the reference camera (world == reference).
    out[reference].placed=true; out[reference].T_world_cam=identity4(); out[reference].hops=0;
    std::queue<int> q; q.push(reference);
    while(!q.empty())
    {
        int a=q.front(); q.pop();
        for (int b : adj[a]) if (!out[b].placed)
        {
            out[b].placed=true;
            out[b].hops=out[a].hops+1;
            out[b].T_world_cam = mat4_mul(out[a].T_world_cam, rel[{a,b}]);   // world←b = world←a · a←b
            q.push(b);
        }
    }

    if (report)
    {
        char buf[512];
        std::string& r=*report; r.clear();
        snprintf(buf,sizeof(buf),"reference camera = %d (world origin)\n",reference); r+=buf;
        for (int b=0;b<N;b++)
        {
            if (b==reference) continue;
            if (!out[b].placed) { snprintf(buf,sizeof(buf),"  cam %d: NOT PLACED (no static-marker path to reference)\n",b); r+=buf; continue; }
            const Mat4& T=out[b].T_world_cam;
            snprintf(buf,sizeof(buf),"  cam %d: placed (%d hop[s]) baseline=%.3f m  origin_in_world=[% .3f % .3f % .3f]\n",
                     b, out[b].hops, std::sqrt(T[3]*T[3]+T[7]*T[7]+T[11]*T[11]), T[3],T[7],T[11]); r+=buf;
        }
        // Cross-camera marker-position agreement: a static marker placed via two
        // different cameras should land at the same world point.
        std::map<int,std::vector<int>> seen_by;
        for (int c=0;c<N;c++) if (out[c].placed) for (auto& kv:cams[c].pose) seen_by[kv.first].push_back(c);
        double max_res=0; int res_marker=-1;
        for (auto& kv : seen_by)
        {
            if (kv.second.size()<2) continue;
            std::vector<std::array<double,3>> w;
            for (int c : kv.second)
            {
                const Mat4& Tcm=cams[c].pose.at(kv.first);
                double org[3]={Tcm[3],Tcm[7],Tcm[11]}, wp[3];
                transform_point(out[c].T_world_cam, org, wp);
                w.push_back({wp[0],wp[1],wp[2]});
            }
            for (size_t i=0;i<w.size();++i) for (size_t j=i+1;j<w.size();++j)
            {
                double dx=w[i][0]-w[j][0],dy=w[i][1]-w[j][1],dz=w[i][2]-w[j][2];
                double d=std::sqrt(dx*dx+dy*dy+dz*dz);
                if (d>max_res){ max_res=d; res_marker=kv.first; }
            }
        }
        if (res_marker>=0)
        { snprintf(buf,sizeof(buf),"  worst cross-camera marker agreement: %.1f mm (marker %d)\n", max_res*1000.0, res_marker); r+=buf; }
    }

    return out;
}

std::array<double,4> mat4_rotation_quat(const Mat4& T)
{
    double R[9] = { T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10] };
    double q[4]; R_to_quat(R, q);
    return { q[0], q[1], q[2], q[3] };
}

std::array<double,3> mat4_transform_point(const Mat4& T, const std::array<double,3>& p)
{
    double pp[3] = { p[0], p[1], p[2] }, o[3];
    transform_point(T, pp, o);
    return { o[0], o[1], o[2] };
}

} // namespace mv
