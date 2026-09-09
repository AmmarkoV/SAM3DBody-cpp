// ════════════════════════════════════════════════════════════════════════════
//  sync_io.cpp — see sync_io.h.
// ════════════════════════════════════════════════════════════════════════════

#include "sync_io.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace mv
{

bool write_sync(const std::string& path, const SyncSession& s)
{
    std::ofstream o(path);
    if (!o) return false;
    o << std::setprecision(15);                  // enough sig-digits for t0_ms etc.

    o << "# SAM3DBody-cpp session.sync v1\n";
    o << "SESSION\n";
    o << "master_fps " << s.master_fps << "\n";

    for (size_t i = 0; i < s.cameras.size(); ++i)
    {
        const SyncCamera& c = s.cameras[i];
        o << "CAMERA " << i << "\n";
        o << "video " << c.video << "\n";
        o << "size " << c.width << " " << c.height << "\n";
        o << "calib " << (c.has_calib ? 1 : 0) << "\n";
        if (c.has_calib)
        {
            o << "K " << c.fx << " " << c.fy << " " << c.cx << " " << c.cy << "\n";
            o << "D " << c.dist[0] << " " << c.dist[1] << " " << c.dist[2]
              << " " << c.dist[3] << " " << c.dist[4] << "\n";
        }
        o << "placed " << (c.placed ? 1 : 0) << "\n";
        if (c.placed)
        {
            o << "T_world_cam";
            for (int k = 0; k < 16; ++k) o << " " << c.T_world_cam[k];
            o << "\n";
        }
        o << "time " << (c.has_time ? 1 : 0) << " " << c.t0_ms << " "
          << c.fps_eff << " " << c.resid_med_ms << "\n";
    }
    return (bool)o;
}

bool read_sync(const std::string& path, SyncSession& s)
{
    std::ifstream in(path);
    if (!in) return false;
    s = SyncSession{};

    std::string line;
    SyncCamera* cur = nullptr;
    while (std::getline(in, line))
    {
        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) continue;          // blank line
        if (key.empty() || key[0] == '#') continue;

        if (key == "SESSION") { continue; }
        else if (key == "master_fps") { ss >> s.master_fps; }
        else if (key == "CAMERA")     { s.cameras.emplace_back(); cur = &s.cameras.back(); }
        else if (!cur)                { continue; }   // keys before any CAMERA: ignore
        else if (key == "video")
        {
            std::string rest;
            std::getline(ss, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            cur->video = rest;
        }
        else if (key == "size")       { ss >> cur->width >> cur->height; }
        else if (key == "calib")      { int v=0; ss >> v; cur->has_calib = (v!=0); }
        else if (key == "K")          { ss >> cur->fx >> cur->fy >> cur->cx >> cur->cy; }
        else if (key == "D")          { for (int k=0;k<5;++k) ss >> cur->dist[k]; }
        else if (key == "placed")     { int v=0; ss >> v; cur->placed = (v!=0); }
        else if (key == "T_world_cam"){ for (int k=0;k<16;++k) ss >> cur->T_world_cam[k]; }
        else if (key == "time")       { int v=0; ss >> v >> cur->t0_ms >> cur->fps_eff >> cur->resid_med_ms;
                                        cur->has_time = (v!=0); }
        // unknown keys are ignored for forward-compatibility
    }
    return true;
}

} // namespace mv
