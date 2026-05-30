// sync_io_test.cpp — write a SyncSession, read it back, check it round-trips.
// Exits 0 on success, 1 on failure.

#include "sync_io.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
static void chk(const char* what, double got, double want, double tol)
{
    if (std::fabs(got - want) > tol) { fprintf(stderr,"  FAIL %-16s %.6f != %.6f\n", what, got, want); g_fail = 1; }
}

int main()
{
    mv::SyncSession s;
    s.master_fps = 59.9401;

    mv::SyncCamera a;
    a.video = "GX010053.MP4"; a.width = 5312; a.height = 2988;
    a.has_calib = true; a.fx = 2629.565; a.fy = 2638.762; a.cx = 2650.711; a.cy = 1469.095;
    a.dist[0]=-0.27885; a.dist[1]=0.10440; a.dist[2]=-0.00062; a.dist[3]=0.00197; a.dist[4]=-0.00725;
    a.placed = true;
    a.T_world_cam = mv::Mat4{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    a.has_time = true; a.t0_ms = 1769516870968.0; a.fps_eff = 59.8802; a.resid_med_ms = 9.5;
    s.cameras.push_back(a);

    mv::SyncCamera b;
    b.video = "DSC_4457.MOV"; b.width = 3840; b.height = 2160;
    b.has_calib = false;                      // approx intrinsics
    b.placed = true;
    b.T_world_cam = mv::Mat4{ 0.7071,-0.7071,0,0.345,  0.7071,0.7071,0,-0.625,  0,0,1,-0.260,  0,0,0,1 };
    b.has_time = true; b.t0_ms = 1769516877724.0; b.fps_eff = 29.9421; b.resid_med_ms = 7.1;
    s.cameras.push_back(b);

    const char* path = "/tmp/sync_io_roundtrip.sync";
    if (!mv::write_sync(path, s)) { fprintf(stderr,"write_sync failed\n"); return 1; }

    mv::SyncSession r;
    if (!mv::read_sync(path, r)) { fprintf(stderr,"read_sync failed\n"); return 1; }

    chk("master_fps", r.master_fps, s.master_fps, 1e-3);
    if (r.cameras.size() != 2) { fprintf(stderr,"  FAIL camera count %zu\n", r.cameras.size()); g_fail=1; }
    else
    {
        const auto& ra = r.cameras[0];
        if (ra.video != "GX010053.MP4") { fprintf(stderr,"  FAIL video '%s'\n", ra.video.c_str()); g_fail=1; }
        chk("a.width", ra.width, 5312, 0.5); chk("a.has_calib", ra.has_calib, 1, 0.5);
        chk("a.fx", ra.fx, 2629.565, 1e-2); chk("a.cy", ra.cy, 1469.095, 1e-2);
        chk("a.d0", ra.dist[0], -0.27885, 1e-5); chk("a.d4", ra.dist[4], -0.00725, 1e-5);
        chk("a.t0", ra.t0_ms, 1769516870968.0, 1.0); chk("a.fps", ra.fps_eff, 59.8802, 1e-3);
        chk("a.placed", ra.placed, 1, 0.5);

        const auto& rb = r.cameras[1];
        if (rb.video != "DSC_4457.MOV") { fprintf(stderr,"  FAIL video '%s'\n", rb.video.c_str()); g_fail=1; }
        chk("b.has_calib", rb.has_calib, 0, 0.5);
        chk("b.T12", rb.T_world_cam[3], 0.345, 1e-4);
        chk("b.T_rot", rb.T_world_cam[0], 0.7071, 1e-4);
        chk("b.t0", rb.t0_ms, 1769516877724.0, 1.0);
    }

    std::remove(path);
    fprintf(stderr, g_fail ? "\nsync_io_test: FAILED\n" : "\nsync_io_test: OK\n");
    return g_fail;
}
