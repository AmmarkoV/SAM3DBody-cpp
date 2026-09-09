// ════════════════════════════════════════════════════════════════════════════
//  libQRSync.c — fullscreen QR timecode display (xqr_time_sync)
//
//  Flashes fullscreen QR codes encoding live Unix timestamps so several cameras
//  pointed at one screen can be aligned to a common clock in post-processing.
//  The multi-view fusion pipeline reads these timecodes back to synchronise
//  streams (see synchronization/README.md and MULTIVIEW_PLAN.md).
//
//  Uses libqrencode for QR generation and X11 + Xrandr for fullscreen output.
//
//  Build:  via the project CMake (target xqr_time_sync), or standalone:
//    gcc -O2 -std=c11 -Wall -Wextra -pedantic libQRSync.c -o xqr_time_sync
//        -lX11 -lXrandr -lqrencode -lm
//  Run:    ./xqr_time_sync --ms --hz 10 --scale 25 --quiet 4 --text --payload "t=%llu"
// ════════════════════════════════════════════════════════════════════════════

#define _POSIX_C_SOURCE 200809L
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/extensions/Xrandr.h>

#include <qrencode.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


// ─── Utilities ───────────────────────────────────────────────────────────────

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static uint64_t now_unix_ms(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) die("clock_gettime failed: %s", strerror(errno));
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

static uint64_t now_unix_s(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) die("clock_gettime failed: %s", strerror(errno));
    return (uint64_t)ts.tv_sec;
}

static uint64_t now_mono_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) die("clock_gettime failed: %s", strerror(errno));
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void sleep_until_ns(uint64_t target_ns)
{
    for (;;)
    {
        uint64_t cur = now_mono_ns();
        if (cur >= target_ns) return;
        uint64_t diff = target_ns - cur;
        struct timespec req;
        req.tv_sec = (time_t)(diff / 1000000000ULL);
        req.tv_nsec = (long)(diff % 1000000000ULL);
        nanosleep(&req, NULL);
    }
}

// ─── Formatting ──────────────────────────────────────────────────────────────

// Format an update rate (Hz) into a compact string (trim trailing zeros).
// Examples: 30   -> "30"
//           29.5 -> "29.5"
//           0.25 -> "0.25"
static void format_hz(double hz, char *out, size_t out_sz)
{
    if (!out || out_sz == 0) return;
    // Start with a reasonably compact representation.
    // %.6g keeps it short while preserving typical CLI-entered values.
    snprintf(out, out_sz, "%.6g", hz);

    // If snprintf chose scientific notation, keep it as-is.
    if (strchr(out, 'e') || strchr(out, 'E')) return;

    // Trim trailing zeros after a decimal point, and then trim the dot.
    char *dot = strchr(out, '.');
    if (!dot) return;
    char *end = out + strlen(out) - 1;
    while (end > dot && *end == '0')
    {
        *end-- = '\0';
    }
    if (end == dot)
    {
        *end = '\0';
    }
}

// Compute a dynamic update rate that oscillates between [min_hz, max_hz]
// in 1 Hz steps, changing every step_T seconds.
// Pattern: min, min+1, ..., max-1, max, max-1, ..., min+1, then repeats.
static double compute_dynamic_hz(int min_hz, int max_hz, double step_T, double elapsed_s)
{
    if (min_hz <= 0) return 1.0;
    if (max_hz < min_hz) max_hz = min_hz;
    if (!(step_T > 0.0 && isfinite(step_T))) step_T = 1.0;
    if (!(elapsed_s >= 0.0 && isfinite(elapsed_s))) elapsed_s = 0.0;

    const int range = max_hz - min_hz;
    if (range == 0) return (double)min_hz;

    const uint64_t step = (uint64_t)floor(elapsed_s / step_T);
    const uint64_t period = (uint64_t)(2 * range);
    const uint64_t p = step % period;

    int hz;
    if ((int)p <= range)
    {
        hz = min_hz + (int)p;
    }
    else
    {
        hz = max_hz - (int)(p - (uint64_t)range);
    }
    if (hz < min_hz) hz = min_hz;
    if (hz > max_hz) hz = max_hz;
    return (double)hz;
}

static int parse_int_hz_arg(const char *s, const char *optname)
{
    if (!s || !*s) die("Missing %s argument", optname);
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) die("Invalid %s value: %s", optname, s);
    double r = round(v);
    if (fabs(v - r) > 1e-9) die("%s must be an integer Hz value (got: %s)", optname, s);
    int iv = (int)r;
    if (iv <= 0) die("%s must be > 0 (got: %d)", optname, iv);
    return iv;
}

// ─── X11 ─────────────────────────────────────────────────────────────────────

typedef struct
{
    Display *dpy;
    int screen;
    Window win;
    GC gc;
    Atom wm_delete;
    int width, height;
    XFontStruct *font;
} XCtx;

static void x11_fullscreen(XCtx *x)
{
    x->screen = DefaultScreen(x->dpy);
    Window root = RootWindow(x->dpy, x->screen);

    // Current screen size via XRandR
    XRRScreenConfiguration *conf = XRRGetScreenInfo(x->dpy, root);
    if (!conf) die("XRandR: XRRGetScreenInfo failed");
    Rotation rot;
    SizeID cur = XRRConfigCurrentConfiguration(conf, &rot);
    int nsizes = 0;
    XRRScreenSize *sizes = XRRConfigSizes(conf, &nsizes);
    if (!sizes || cur >= (SizeID)nsizes) die("XRandR: could not get current screen size");
    x->width = sizes[cur].width;
    x->height = sizes[cur].height;
    XRRFreeScreenConfigInfo(conf);

    XSetWindowAttributes swa;
    swa.override_redirect = True;
    swa.background_pixel = BlackPixel(x->dpy, x->screen);

    x->win = XCreateWindow(
                 x->dpy, root,
                 0, 0, (unsigned)x->width, (unsigned)x->height,
                 0, CopyFromParent, InputOutput, CopyFromParent,
                 CWOverrideRedirect | CWBackPixel, &swa);

    XSelectInput(x->dpy, x->win, ExposureMask | KeyPressMask | StructureNotifyMask);

    x->gc = XCreateGC(x->dpy, x->win, 0, NULL);

    x->wm_delete = XInternAtom(x->dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(x->dpy, x->win, &x->wm_delete, 1);

    // Load a readable font (fallback to default GC font if this fails)
    x->font = XLoadQueryFont(x->dpy, "9x15");
    if (!x->font) x->font = XLoadQueryFont(x->dpy, "fixed");
    if (x->font) XSetFont(x->dpy, x->gc, x->font->fid);

    XMapRaised(x->dpy, x->win);
    XSync(x->dpy, False);

    // Force focus + grab keyboard so Esc works reliably
    XSetInputFocus(x->dpy, x->win, RevertToParent, CurrentTime);
    int gk = XGrabKeyboard(x->dpy, x->win, True, GrabModeAsync, GrabModeAsync, CurrentTime);
    if (gk != GrabSuccess)
    {
        fprintf(stderr, "Warning: XGrabKeyboard failed (%d). Esc/q may not work if window lacks focus.\n", gk);
    }
    XSync(x->dpy, False);
}

static QRecLevel ecc_from_char(char c)
{
    switch (c)
    {
    case 'L':
    case 'l':
        return QR_ECLEVEL_L;
    case 'M':
    case 'm':
        return QR_ECLEVEL_M;
    case 'Q':
    case 'q':
        return QR_ECLEVEL_Q;
    case 'H':
    case 'h':
        return QR_ECLEVEL_H;
    default:
        return QR_ECLEVEL_M;
    }
}


static void draw_qr_at(XCtx *x, const QRcode *qr,
                       int scale, int quiet, bool invert,
                       int ox, int oy)
{
    const int qrSize = qr->width;
    const int total = qrSize + 2 * quiet;
    const int pix = scale;

    const int imgW = total * pix;
    const int imgH = total * pix;

    unsigned long black = BlackPixel(x->dpy, x->screen);
    unsigned long white = WhitePixel(x->dpy, x->screen);

    // QR background
    XSetForeground(x->dpy, x->gc, invert ? black : white);
    XFillRectangle(x->dpy, x->win, x->gc, ox, oy, (unsigned)imgW, (unsigned)imgH);

    // Modules
    XSetForeground(x->dpy, x->gc, invert ? white : black);
    const unsigned char *data = qr->data;

    for (int y = 0; y < qrSize; y++)
    {
        for (int xx = 0; xx < qrSize; xx++)
        {
            int v = data[y * qrSize + xx] & 1; // libqrencode: LSB=black
            if (!v) continue;
            int px0 = ox + (xx + quiet) * pix;
            int py0 = oy + (y + quiet) * pix;
            XFillRectangle(x->dpy, x->win, x->gc, px0, py0, (unsigned)pix, (unsigned)pix);
        }
    }
}

static void draw_qr_scene(XCtx *x, const QRcode *qr,
                          int scale, int quiet, bool invert,
                          int tile_x, bool have_setx, int setx,
                          bool have_sety, int sety,
                          bool draw_text, const char *text_line)
{
    const int qrSize = qr->width;
    const int total = qrSize + 2 * quiet;
    const int pix = scale;

    const int imgW = total * pix;
    const int imgH = total * pix;

    if (tile_x < 1) tile_x = 1;

    unsigned long black = BlackPixel(x->dpy, x->screen);
    unsigned long white = WhitePixel(x->dpy, x->screen);

    // Full background
    XSetForeground(x->dpy, x->gc, invert ? white : black);
    XFillRectangle(x->dpy, x->win, x->gc, 0, 0, (unsigned)x->width, (unsigned)x->height);

    // Treat the full X screen as tile_x equal-width "segments" (handy for a 1x3 Xinerama/XRandR desktop).
    int segW = x->width / tile_x;
    if (segW <= 0) segW = x->width;

    int oy = have_sety ? sety : (x->height - imgH) / 2;

    for (int i = 0; i < tile_x; i++)
    {
        int baseX = i * segW;
        int ox = have_setx ? (baseX + setx) : (baseX + (segW - imgW) / 2);
        draw_qr_at(x, qr, scale, quiet, invert, ox, oy);
    }

    // Always-visible text: white on black with a small backing box
    if (draw_text && text_line)
    {
        int margin = 16;
        int tx = margin;
        int ty = margin + 20;

        // Compute text extents if we have a font
        int tw = (int)strlen(text_line) * 8;
        int th = 18;
        if (x->font)
        {
            tw = XTextWidth(x->font, text_line, (int)strlen(text_line));
            th = x->font->ascent + x->font->descent + 6;
            ty = margin + x->font->ascent + 3;
        }

        // Box
        XSetForeground(x->dpy, x->gc, invert ? white : black);
        XFillRectangle(x->dpy, x->win, x->gc,
                       tx - 8, ty - (th - 6), (unsigned)(tw + 16), (unsigned)th);

        // Text
        XSetForeground(x->dpy, x->gc, invert ? black : white);
        XDrawString(x->dpy, x->win, x->gc, tx, ty, text_line, (int)strlen(text_line));
    }

    XSync(x->dpy, False);
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

typedef struct
{
    double hz;
    bool dhz_enabled;
    int dhz_min_hz;
    int dhz_max_hz;
    double dhz_step_T;
    bool use_ms;
    char ec;          // 'L','M','Q','H'
    int scale;
    int quiet;
    bool invert;
    bool draw_text;
    char payload_fmt[256];
    bool payload_specified;
    bool raw_only;
    int tile_x;
    bool have_setx;
    bool have_sety;
    int setx;
    int sety;
} Opts;

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "Options:\n"
            "  --hz <float>          Update rate (default 30)\n"
            "  --dhz <min> <max> <T> Dynamic Hz: oscillate min->max->min in 1Hz steps every T seconds\n"
            "  --ms | --sec          Encode unix time in milliseconds (default) or seconds\n"
            "  --ec <L|M|Q|H>         QR error correction (default M)\n"
            "  --scale <int>         Pixels per module (default 16)\n"
            "  --quiet <int>         Quiet zone modules (default 4)\n"
            "  --invert              Invert colors\n"
            "  --text                Draw payload text (always visible)\n"
            "  --payload <fmt>       printf-style payload format (default: hz=<hz>&t=%%llu)\n"
            "  --raw                 Payload is ONLY the timestamp number\n"
            "  --help                Show this help\n"
            "Controls:\n"
            "  Space: pause/resume\n"
            "  Esc/q: quit\n",
            argv0);
}

static Opts parse_args(int argc, char **argv)
{
    Opts o= {0};
    o.hz = 30.0;
    o.dhz_enabled = false;
    o.dhz_min_hz = 0;
    o.dhz_max_hz = 0;
    o.dhz_step_T = 0.0;
    o.use_ms = true;
    o.ec = 'M';
    o.scale = 16;
    o.quiet = 4;
    o.invert = false;
    o.draw_text = false;
    o.raw_only = false;
    o.tile_x = 1;
    o.have_setx = false;
    o.have_sety = false;
    o.setx = 0;
    o.sety = 0;
    snprintf(o.payload_fmt, sizeof(o.payload_fmt), "t=%%llu");
    o.payload_specified = false;

    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0)
        {
            usage(argv[0]);
            exit(0);
        }
        else
        if (strcmp(a, "--hz") == 0 && i + 1 < argc)
        {
            o.hz = atof(argv[++i]);
            if (!(o.hz > 0.0 && isfinite(o.hz))) die("Invalid --hz");
        }
        else
        if (strcmp(a, "--dhz") == 0 && i + 3 < argc)
        {
            o.dhz_min_hz = parse_int_hz_arg(argv[++i], "--dhz minHz");
            o.dhz_max_hz = parse_int_hz_arg(argv[++i], "--dhz maxHz");
            o.dhz_step_T = atof(argv[++i]);
            if (!(isfinite(o.dhz_step_T) && o.dhz_step_T > 0.0)) die("Invalid --dhz T (must be > 0)");
            if (o.dhz_max_hz < o.dhz_min_hz)
            {
                int t = o.dhz_min_hz;
                o.dhz_min_hz = o.dhz_max_hz;
                o.dhz_max_hz = t;
            }
            o.dhz_enabled = true;
        }
        else
        if (strcmp(a, "--ms") == 0)
        {
            o.use_ms = true;
        }
        else
        if (strcmp(a, "--sec") == 0)
        {
            o.use_ms = false;
        }
        else
        if (strcmp(a, "--ec") == 0 && i + 1 < argc)
        {
            o.ec = argv[++i][0];
        }
        else
        if (strcmp(a, "--scale") == 0 && i + 1 < argc)
        {
            o.scale = atoi(argv[++i]);
            if (o.scale <= 0) die("Invalid --scale");
        }
        else
        if (strcmp(a, "--quiet") == 0 && i + 1 < argc)
        {
            o.quiet = atoi(argv[++i]);
            if (o.quiet < 0) die("Invalid --quiet");
        }
        else
        if (strcmp(a, "--invert") == 0)
        {
            o.invert = true;
        }
        else
        if (strcmp(a, "--text") == 0)
        {
            o.draw_text = true;
        }
        else
        if (strcmp(a, "--payload") == 0 && i + 1 < argc)
        {
            snprintf(o.payload_fmt, sizeof(o.payload_fmt), "%s", argv[++i]);
            o.payload_specified = true;
        }
        else
        if (strcmp(a, "--raw") == 0)
        {
            o.raw_only = true;
        }
        else
        if (strcmp(a, "--tile") == 0 && i + 1 < argc)
        {
            o.tile_x = atoi(argv[++i]);
            if (o.tile_x < 1) die("Invalid --tile");
        }
        else
        if (strcmp(a, "--setX") == 0 && i + 1 < argc)
        {
            o.setx = atoi(argv[++i]);
            o.have_setx = true;
        }
        else
        if (strcmp(a, "--setY") == 0 && i + 1 < argc)
        {
            o.sety = atoi(argv[++i]);
            o.have_sety = true;
        }
        else
        {
            die("Unknown option: %s (try --help)", a);
        }
    }

    if (o.payload_fmt[0] == '\0')
    {
        snprintf(o.payload_fmt, sizeof(o.payload_fmt), "t=%%llu");
    }

    return o;
}

int main(int argc, char **argv)
{
    Opts opt = parse_args(argc, argv);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) die("XOpenDisplay failed. Is DISPLAY set?");

    XCtx x = {0};
    x.dpy = dpy;
    x11_fullscreen(&x);

    bool running = true;
    bool paused = false;
    uint64_t frame = 0;

    // Scheduling is based on a monotonic clock so dynamic Hz stays stable even if wall clock changes.
    const uint64_t start_ns = now_mono_ns();
    double init_hz = opt.dhz_enabled
                     ? compute_dynamic_hz(opt.dhz_min_hz, opt.dhz_max_hz, opt.dhz_step_T, 0.0)
                     : opt.hz;
    uint64_t interval_ns = (uint64_t)llround(1e9 / init_hz);
    if (interval_ns == 0) interval_ns = 1;
    uint64_t next_ns = start_ns;

    while (running)
    {
        while (XPending(x.dpy))
        {
            XEvent ev;
            XNextEvent(x.dpy, &ev);
            if (ev.type == KeyPress)
            {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_Escape || ks == XK_q)
                {
                    running = false;
                }
                else
                if (ks == XK_space)
                {
                    paused = !paused;
                }
            }
            else
            if (ev.type == ClientMessage)
            {
                if ((Atom)ev.xclient.data.l[0] == x.wm_delete) running = false;
            }
            else
            if (ev.type == ConfigureNotify)
            {
                x.width = ev.xconfigure.width;
                x.height = ev.xconfigure.height;
            }
        }

        const uint64_t cur_ns = now_mono_ns();
        const double elapsed_s = (double)(cur_ns - start_ns) / 1e9;
        const double cur_hz = opt.dhz_enabled
                              ? compute_dynamic_hz(opt.dhz_min_hz, opt.dhz_max_hz, opt.dhz_step_T, elapsed_s)
                              : opt.hz;

        if (!paused)
        {
            uint64_t t = opt.use_ms ? now_unix_ms() : now_unix_s();

            char payload[512];
            if (opt.raw_only)
            {
                snprintf(payload, sizeof(payload), "%" PRIu64, t);
            }
            else
            if (!opt.payload_specified)
            {
                // Default behavior: include update rate and timestamp.
                // Result: hz=<hz>&t=<unix_time>
                char hzbuf[32];
                format_hz(cur_hz, hzbuf, sizeof(hzbuf));
                snprintf(payload, sizeof(payload), "hz=%s&t=%llu", hzbuf, (unsigned long long)t);
            }
            else
            {
                snprintf(payload, sizeof(payload), opt.payload_fmt, (unsigned long long)t, (unsigned long long)frame);
            }

            // Encode QR (8-bit, auto version, specified ECC)
            QRecLevel level = ecc_from_char(opt.ec);
            QRcode *qr = QRcode_encodeString8bit(payload, 0, level);
            if (!qr)
            {
                // fallback to raw timestamp if format too long or error
                snprintf(payload, sizeof(payload), "%" PRIu64, t);
                qr = QRcode_encodeString8bit(payload, 0, level);
                if (!qr) die("QRcode_encodeString8bit failed");
            }

            draw_qr_scene(&x, qr, opt.scale, opt.quiet, opt.invert,
                          opt.tile_x, opt.have_setx, opt.setx, opt.have_sety, opt.sety,
                          opt.draw_text, payload);
            QRcode_free(qr);

            frame++;
        }

        interval_ns = (uint64_t)llround(1e9 / cur_hz);
        if (interval_ns == 0) interval_ns = 1;
        next_ns += interval_ns;
        sleep_until_ns(next_ns);
    }

    XUngrabKeyboard(x.dpy, CurrentTime);
    if (x.font) XFreeFont(x.dpy, x.font);
    XDestroyWindow(x.dpy, x.win);
    XCloseDisplay(x.dpy);
    return 0;
}

