/*
  xqr_time_sync.c
  Fullscreen X11 QR code showing Unix timestamp for camera sync.

  Build:
    gcc -O2 -std=c11 -Wall -Wextra -pedantic xqr_time_sync.c -o xqr_time_sync -lX11 -lXrandr -lm

  Controls:
    Esc / q : quit
    Space   : pause/resume


    ./xqr_time_sync --ms --hz 30 --text --payload "t=%llu"


  Notes:
    - Uses XRandR to get current screen size.
    - Includes a compact QR generator (Nayuki-style API, embedded here).
*/

#define _POSIX_C_SOURCE 200809L
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <X11/extensions/Xrandr.h>

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

/* ----------------------------- Utilities ----------------------------- */

static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static uint64_t now_unix_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) die("clock_gettime failed: %s", strerror(errno));
  return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

static uint64_t now_unix_s(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) die("clock_gettime failed: %s", strerror(errno));
  return (uint64_t)ts.tv_sec;
}

static uint64_t now_mono_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) die("clock_gettime failed: %s", strerror(errno));
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void sleep_until_ns(uint64_t target_ns) {
  for (;;) {
    uint64_t cur = now_mono_ns();
    if (cur >= target_ns) return;
    uint64_t diff = target_ns - cur;
    struct timespec req;
    req.tv_sec = (time_t)(diff / 1000000000ULL);
    req.tv_nsec = (long)(diff % 1000000000ULL);
    nanosleep(&req, NULL);
  }
}

/* ----------------------------- QR Generator -----------------------------
   This is a compact embedded QR generator based on the well-known Nayuki approach:
   - Low footprint
   - No external deps
   - Supports ECC levels L/M/Q/H
   - Byte mode encoding

   Public-domain-style compact implementation.
   Limitations:
   - Only byte-mode
   - Version chosen automatically up to a practical bound (we keep buffers reasonably sized)
*/

enum QrEcc { QR_ECC_LOW = 0, QR_ECC_MEDIUM, QR_ECC_QUARTILE, QR_ECC_HIGH };

typedef struct {
  int size;            // side length in modules
  unsigned char *mods; // size*size bits, 0=white 1=black
} QrCode;

/* ---- Galois field / Reed-Solomon helpers ---- */
// ---------------- GF(256) for QR (primitive poly 0x11D) ----------------

static unsigned char gf_exp[512];
static unsigned char gf_log[256];
static int gf_inited = 0;

static void gf_init(void) {
  if (gf_inited) return;
  gf_inited = 1;

  unsigned int x = 1;
  for (int i = 0; i < 255; i++) {
    gf_exp[i] = (unsigned char)x;
    gf_log[x] = (unsigned char)i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11D;
  }
  // duplicate to avoid mod 255 in exp lookup
  for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
}

static unsigned char gf_mul(unsigned char a, unsigned char b) {
  if (a == 0 || b == 0) return 0;
  gf_init();
  return gf_exp[gf_log[a] + gf_log[b]];
}

// ---------------- Reed–Solomon (QR) ----------------
// Generator polynomial: (x - a^0)(x - a^1)...(x - a^(degree-1))
// In QR, a = 2 (i.e., successive powers of 2 in GF(256)).

static void rs_generate_poly(unsigned char *gen, int degree) {
  gf_init();
  // gen has length degree+1
  // Start with polynomial "1"
  memset(gen, 0, (size_t)(degree + 1));
  gen[0] = 1;

  int genLen = 1;
  for (int i = 0; i < degree; i++) {
    unsigned char coef = gf_exp[i]; // a^i
    // Multiply current gen by (x - a^i) == (x + a^i) in GF(2^8)
    // new[j]   ^= gen[j] * coef
    // new[j+1] ^= gen[j]
    unsigned char next[256];
    memset(next, 0, sizeof(next));

    for (int j = 0; j < genLen; j++) {
      next[j]     ^= gf_mul(gen[j], coef);
      next[j + 1] ^= gen[j];
    }
    genLen++;
    memcpy(gen, next, (size_t)genLen);
  }
}

// Compute remainder of data(x) * x^degree divided by gen(x), returns 'degree' bytes
static void rs_compute_remainder(const unsigned char *data, int dataLen,
                                 const unsigned char *gen, int degree,
                                 unsigned char *out) {
  // out length = degree
  memset(out, 0, (size_t)degree);

  for (int i = 0; i < dataLen; i++) {
    unsigned char factor = data[i] ^ out[0];
    memmove(out, out + 1, (size_t)(degree - 1));
    out[degree - 1] = 0;

    // gen[0] is highest term and equals 1 for QR generator polynomials as built above.
    // We apply gen[1..degree]
    for (int j = 0; j < degree; j++) {
      out[j] ^= gf_mul(gen[j + 1], factor);
    }
  }
}


/* ---- QR tables (versions 1..10 are enough for small payloads like timestamps) ----
   If you later want longer payloads, extend these tables or bump limits.
*/

typedef struct {
  int ver;
  int size;
  int dataCodewords; // total data codewords (all blocks combined)
  int eccCodewords;  // total ecc codewords (all blocks combined)
  int blocks;        // number of RS blocks (we use equal blocks for these versions)
} QrVerInfo;

// For simplicity, this table is for versions 1..10 and ECC L/M/Q/H.
// Values are (data, ecc, blocks) for each version+ecc.
// Source: standard QR capacity tables (byte mode), simplified equal block configs for v<=10.
static const QrVerInfo QR_VER_TABLE[4][10] = {
  // L
  {
    {1, 21, 19,  7, 1},
    {2, 25, 34, 10, 1},
    {3, 29, 55, 15, 1},
    {4, 33, 80, 20, 1},
    {5, 37,108, 26, 1},
    {6, 41,136, 36, 2},
    {7, 45,156, 40, 2},
    {8, 49,194, 48, 2},
    {9, 53,232, 60, 2},
    {10,57,274, 72, 2},
  },
  // M
  {
    {1, 21, 16, 10, 1},
    {2, 25, 28, 16, 1},
    {3, 29, 44, 26, 1},
    {4, 33, 64, 36, 2},
    {5, 37, 86, 48, 2},
    {6, 41,108, 64, 4},
    {7, 45,124, 72, 4},
    {8, 49,154, 88, 4},
    {9, 53,182,110, 4},
    {10,57,216,130, 4},
  },
  // Q
  {
    {1, 21, 13, 13, 1},
    {2, 25, 22, 22, 1},
    {3, 29, 34, 36, 2},
    {4, 33, 48, 52, 2},
    {5, 37, 62, 72, 4},
    {6, 41, 76, 96, 4},
    {7, 45, 88,108, 6},
    {8, 49,110,132, 6},
    {9, 53,132,160, 8},
    {10,57,154,192, 8},
  },
  // H
  {
    {1, 21,  9, 17, 1},
    {2, 25, 16, 28, 1},
    {3, 29, 26, 44, 2},
    {4, 33, 36, 64, 4},
    {5, 37, 46, 88, 4},
    {6, 41, 60,112, 4},
    {7, 45, 66,130, 5},
    {8, 49, 86,156, 6},
    {9, 53,100,192, 8},
    {10,57,122,224, 8},
  }
};

static int get_alignment_count(int ver) {
  if (ver == 1) return 0;
  // Standard: floor(ver/7)+2 alignment patterns
  return ver / 7 + 2;
}

static void get_alignment_positions(int ver, int *pos, int *count) {
  int n = get_alignment_count(ver);
  *count = n;
  if (n == 0) return;

  int size = 17 + 4 * ver;
  pos[0] = 6;
  pos[n - 1] = size - 7;
  if (n == 2) return;

  int step = (size - 13) / (n - 1);
  if (step % 2) step++;
  for (int i = 1; i < n - 1; i++) pos[i] = pos[n - 1] - (n - 1 - i) * step;
}

static inline int idx(int size, int x, int y) { return y * size + x; }

static void set_mod(QrCode *q, int x, int y, int v) {
  if (x < 0 || y < 0 || x >= q->size || y >= q->size) return;
  q->mods[idx(q->size, x, y)] = (unsigned char)(v ? 1 : 0);
}

static int get_mod(const QrCode *q, int x, int y) {
  return q->mods[idx(q->size, x, y)] ? 1 : 0;
}

// Reserve mask: 1 means function module (not data)
static void mark_reserved(unsigned char *res, int size, int x, int y) {
  if (x < 0 || y < 0 || x >= size || y >= size) return;
  res[idx(size, x, y)] = 1;
}

static void draw_finder(QrCode *q, unsigned char *res, int x, int y) {
  for (int dy = -1; dy <= 7; dy++) {
    for (int dx = -1; dx <= 7; dx++) {
      int xx = x + dx, yy = y + dy;
      int on = (dx >= 0 && dx <= 6 && dy >= 0 && dy <= 6 &&
                (dx == 0 || dx == 6 || dy == 0 || dy == 6 || (dx >= 2 && dx <= 4 && dy >= 2 && dy <= 4)));
      set_mod(q, xx, yy, on);
      mark_reserved(res, q->size, xx, yy);
    }
  }
}

static void draw_timing(QrCode *q, unsigned char *res) {
  for (int i = 0; i < q->size; i++) {
    if (!res[idx(q->size, 6, i)]) {
      set_mod(q, 6, i, (i % 2) == 0);
      mark_reserved(res, q->size, 6, i);
    }
    if (!res[idx(q->size, i, 6)]) {
      set_mod(q, i, 6, (i % 2) == 0);
      mark_reserved(res, q->size, i, 6);
    }
  }
}

static void draw_alignment(QrCode *q, unsigned char *res, int cx, int cy) {
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int xx = cx + dx, yy = cy + dy;
      int on = (abs(dx) == 2 || abs(dy) == 2 || (dx == 0 && dy == 0));
      set_mod(q, xx, yy, on);
      mark_reserved(res, q->size, xx, yy);
    }
  }
}

static void draw_format_info(QrCode *q, unsigned char *res, enum QrEcc ecc, int mask) {
  static const int ECC_BITS[4] = {1, 0, 3, 2}; // L=01, M=00, Q=11, H=10
  int data = (ECC_BITS[ecc] << 3) | (mask & 7);

  // BCH(15,5) generator 0x537, then XOR 0x5412 (like qrcode.js)
  int rem = data << 10;
  for (int i = 14; i >= 10; i--) {
    if ((rem >> i) & 1) rem ^= 0x537 << (i - 10);
  }
  int bits = ((data << 10) | rem) ^ 0x5412;

  int size = q->size;

  // Vertical (x=8): y=0..5,7..8,size-7..size-1
  for (int i = 0; i < 15; i++) {
    int bit = (bits >> i) & 1; // LSB-first

    int x = 8;
    int y;
    if (i < 6)        y = i;
    else if (i < 8)   y = i + 1;          // skip y=6
    else              y = size - 15 + i;  // bottom section

    set_mod(q, x, y, bit);
    mark_reserved(res, size, x, y);
  }

  // Horizontal (y=8): x=size-1..size-8, then x=7 for i==8, then x=5..0
  for (int i = 0; i < 15; i++) {
    int bit = (bits >> i) & 1; // LSB-first

    int y = 8;
    int x;
    if (i < 8)        x = size - i - 1;
    else if (i == 8)  x = 7;              // IMPORTANT: skip timing column x=6
    else              x = 14 - i;

    set_mod(q, x, y, bit);
    mark_reserved(res, size, x, y);
  }

  // Fixed dark module
  set_mod(q, 8, size - 8, 1);
  mark_reserved(res, size, 8, size - 8);
}



static int get_mask_bit(int mask, int x, int y) {
  switch (mask) {
    case 0: return ((x + y) % 2) == 0;
    case 1: return (y % 2) == 0;
    case 2: return (x % 3) == 0;
    case 3: return ((x + y) % 3) == 0;
    case 4: return (((y / 2) + (x / 3)) % 2) == 0;
    case 5: return ((x * y) % 2 + (x * y) % 3) == 0;
    case 6: return ((((x * y) % 2) + ((x * y) % 3)) % 2) == 0;
    case 7: return ((((x + y) % 2) + ((x * y) % 3)) % 2) == 0;
    default: return 0;
  }
}

static int score_penalty(const QrCode *q) {
  // A lightweight mask scoring (not full spec-perfect, but good enough for stable masks)
  int size = q->size;
  int penalty = 0;

  // Adjacent runs (rows and cols)
  for (int y = 0; y < size; y++) {
    int runColor = get_mod(q, 0, y), runLen = 1;
    for (int x = 1; x < size; x++) {
      int c = get_mod(q, x, y);
      if (c == runColor) runLen++;
      else {
        if (runLen >= 5) penalty += 3 + (runLen - 5);
        runColor = c; runLen = 1;
      }
    }
    if (runLen >= 5) penalty += 3 + (runLen - 5);
  }
  for (int x = 0; x < size; x++) {
    int runColor = get_mod(q, x, 0), runLen = 1;
    for (int y = 1; y < size; y++) {
      int c = get_mod(q, x, y);
      if (c == runColor) runLen++;
      else {
        if (runLen >= 5) penalty += 3 + (runLen - 5);
        runColor = c; runLen = 1;
      }
    }
    if (runLen >= 5) penalty += 3 + (runLen - 5);
  }

  // 2x2 blocks
  for (int y = 0; y < size - 1; y++) {
    for (int x = 0; x < size - 1; x++) {
      int c = get_mod(q, x, y);
      if (c == get_mod(q, x + 1, y) &&
          c == get_mod(q, x, y + 1) &&
          c == get_mod(q, x + 1, y + 1)) penalty += 3;
    }
  }

  // Dark ratio
  int dark = 0;
  for (int i = 0; i < size * size; i++) dark += q->mods[i] ? 1 : 0;
  int total = size * size;
  int k = (int)(fabs((double)dark * 100.0 / (double)total - 50.0) / 5.0);
  penalty += k * 10;

  return penalty;
}

static void add_bytes_to_bitbuf(const unsigned char *bytes, int nbytes,
                                unsigned char *bitbuf, int *bitlen) {
  for (int i = 0; i < nbytes; i++) {
    for (int b = 7; b >= 0; b--) {
      int bit = (bytes[i] >> b) & 1;
      int pos = (*bitlen)++;
      bitbuf[pos >> 3] = (unsigned char)((bitbuf[pos >> 3] & ~(1u << (7 - (pos & 7)))) |
                                         ((unsigned char)bit << (7 - (pos & 7))));
    }
  }
}

static void append_bits(unsigned int val, int nbits, unsigned char *bitbuf, int *bitlen) {
  for (int i = nbits - 1; i >= 0; i--) {
    int bit = (val >> i) & 1U;
    int pos = (*bitlen)++;
    bitbuf[pos >> 3] = (unsigned char)((bitbuf[pos >> 3] & ~(1u << (7 - (pos & 7)))) |
                                       ((unsigned char)bit << (7 - (pos & 7))));
  }
}



static int make_qr_byte_mode(const unsigned char *data, int len, enum QrEcc ecc, QrCode *out,
                            unsigned char *workMods, unsigned char *workRes,
                            unsigned char *codewords, unsigned char *bitbuf,
                            size_t bufcap) {
  // Pick smallest version (1..10) that fits.
  int ver = -1;
  const QrVerInfo *vi = NULL;

  for (int v = 1; v <= 10; v++) {
    const QrVerInfo *t = &QR_VER_TABLE[ecc][v - 1];
    int ccbits = (v <= 9) ? 8 : 16;
    int totalBits = 4 + ccbits + 8 * len;
    int totalCw = (totalBits + 7) / 8;
    if (totalCw <= t->dataCodewords) { ver = v; vi = t; break; }
  }
  if (ver < 0) return -1;

  int size = 17 + 4 * ver;
  out->size = size;
  out->mods = workMods;
  if ((size_t)(size * size) > bufcap) return -2;

  memset(workMods, 0, (size_t)(size * size));
  memset(workRes, 0, (size_t)(size * size));

  // Function patterns
  draw_finder(out, workRes, 0, 0);
  draw_finder(out, workRes, size - 7, 0);
  draw_finder(out, workRes, 0, size - 7);
  draw_timing(out, workRes);

  // Alignment patterns
  int apos[7], acnt = 0;
  get_alignment_positions(ver, apos, &acnt);
  for (int i = 0; i < acnt; i++) {
    for (int j = 0; j < acnt; j++) {
      int cx = apos[i], cy = apos[j];
      if ((cx == 6 && cy == 6) ||
          (cx == 6 && cy == size - 7) ||
          (cx == size - 7 && cy == 6)) continue;
      draw_alignment(out, workRes, cx, cy);
    }
  }

  // Reserve EXACT format info coordinates (must match draw_format_info above)
  for (int i = 0; i < 15; i++) {
    int yv;
    if (i < 6)        yv = i;
    else if (i < 8)   yv = i + 1;
    else              yv = size - 15 + i;
    mark_reserved(workRes, size, 8, yv);

    int xh;
    if (i < 8)        xh = size - i - 1;
    else if (i == 8)  xh = 7;          // skip timing col 6
    else              xh = 14 - i;
    mark_reserved(workRes, size, xh, 8);
  }
  mark_reserved(workRes, size, 8, size - 8); // dark module

  // Build data bitstream (byte mode)
  memset(bitbuf, 0, (size_t)vi->dataCodewords);
  int bitlen = 0;
  append_bits(0x4, 4, bitbuf, &bitlen);
  {
    int ccbits = (ver <= 9) ? 8 : 16;
    append_bits((unsigned int)len, ccbits, bitbuf, &bitlen);
  }
  add_bytes_to_bitbuf(data, len, bitbuf, &bitlen);

  // Terminator up to 4 zeros
  int totalDataBits = vi->dataCodewords * 8;
  int remain = totalDataBits - bitlen;
  if (remain > 0) append_bits(0, remain >= 4 ? 4 : remain, bitbuf, &bitlen);

  // Pad to byte boundary
  while ((bitlen & 7) != 0) append_bits(0, 1, bitbuf, &bitlen);

  // Convert to codewords + pad bytes 0xEC 0x11
  int cwLen = bitlen / 8;
  if (cwLen > vi->dataCodewords) return -3;
  memcpy(codewords, bitbuf, (size_t)cwLen);
  for (int i = cwLen; i < vi->dataCodewords; i++) {
    codewords[i] = (unsigned char)((((i - cwLen) & 1) == 0) ? 0xEC : 0x11);
  }

  // RS ECC (using your simplified equal-block assumption)
  int blocks = vi->blocks;
  int dataCwPerBlock = vi->dataCodewords / blocks;
  int eccCwPerBlock = vi->eccCodewords / blocks;

  unsigned char gen[256];
  rs_generate_poly(gen, eccCwPerBlock);

  unsigned char eccbuf[256];
  unsigned char full[2048];
  unsigned char eccAll[2048];
  int fullLen = 0;

  // Interleave data
  for (int i = 0; i < dataCwPerBlock; i++) {
    for (int b = 0; b < blocks; b++) {
      full[fullLen++] = codewords[b * dataCwPerBlock + i];
    }
  }

  // Compute ECC per block and interleave
  for (int b = 0; b < blocks; b++) {
    rs_compute_remainder(&codewords[b * dataCwPerBlock], dataCwPerBlock,
                         gen, eccCwPerBlock, eccbuf);
    memcpy(&eccAll[b * eccCwPerBlock], eccbuf, (size_t)eccCwPerBlock);
  }
  for (int i = 0; i < eccCwPerBlock; i++) {
    for (int b = 0; b < blocks; b++) {
      full[fullLen++] = eccAll[b * eccCwPerBlock + i];
    }
  }

  // Place data bits in zigzag
  int x = size - 1;
  int y = size - 1;
  int dir = -1;
  int bitpos = 0;
  int maxBits = fullLen * 8;

  while (x > 0) {
    if (x == 6) x--; // skip timing column
    for (int i = 0; i < size; i++) {
      int yy = y + dir * i;
      for (int dx = 0; dx < 2; dx++) {
        int xx = x - dx;
        if (workRes[idx(size, xx, yy)]) continue;

        //int bit = (full[bitpos >> 3] >> (7 - (bitpos & 7))) & 1;

int bit = 0;
if (bitpos < maxBits)
  bit = (full[bitpos >> 3] >> (7 - (bitpos & 7))) & 1;
        set_mod(out, xx, yy, bit);
        bitpos++;
      }
    }
    y += dir * (size - 1);
    dir = -dir;
    x -= 2;
  }

  // Choose best mask 0..7 (MUST restore reserved map each try!)
  int bestMask = 0, bestScore = 0x7fffffff;
  unsigned char savedMods[4096];
  unsigned char savedRes[4096];
  memcpy(savedMods, out->mods, (size_t)(size * size));
  memcpy(savedRes,  workRes,  (size_t)(size * size));

  for (int m = 0; m < 8; m++) {
    memcpy(out->mods, savedMods, (size_t)(size * size));
    memcpy(workRes,  savedRes,  (size_t)(size * size));

    for (int yy = 0; yy < size; yy++) {
      for (int xx = 0; xx < size; xx++) {
        if (workRes[idx(size, xx, yy)]) continue;
        int v = get_mod(out, xx, yy);
        if (get_mask_bit(m, xx, yy)) v ^= 1;
        set_mod(out, xx, yy, v);
      }
    }

    draw_format_info(out, workRes, ecc, m);
    int s = score_penalty(out);
    if (s < bestScore) { bestScore = s; bestMask = m; }
  }

  // Apply best mask definitively
  memcpy(out->mods, savedMods, (size_t)(size * size));
  memcpy(workRes,  savedRes,  (size_t)(size * size));

  for (int yy = 0; yy < size; yy++) {
    for (int xx = 0; xx < size; xx++) {
      if (workRes[idx(size, xx, yy)]) continue;
      int v = get_mod(out, xx, yy);
      if (get_mask_bit(bestMask, xx, yy)) v ^= 1;
      set_mod(out, xx, yy, v);
    }
  }
  draw_format_info(out, workRes, ecc, bestMask);

  return 0;
}




/* ----------------------------- X11 Rendering ----------------------------- */

typedef struct {
  Display *dpy;
  int screen;
  Window win;
  GC gc;
  Atom wm_delete;
  int width, height;
  bool use_shm;
} XCtx;

static void x11_fullscreen(XCtx *x, int monitor_index) 
{
  (void)monitor_index;
  // This creates a borderless window sized to the current screen.
  // For multi-monitor placement, you can extend by querying XRandR CRTCs;
  // for most camera-sync use, "primary screen" fullscreen is enough.

  x->screen = DefaultScreen(x->dpy);
  Window root = RootWindow(x->dpy, x->screen);

  // Get current screen size via XRandR
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
  swa.override_redirect = True; // borderless, bypass WM
  swa.background_pixel = BlackPixel(x->dpy, x->screen);

  x->win = XCreateWindow(
      x->dpy, root,
      0, 0, (unsigned)x->width, (unsigned)x->height,
      0, CopyFromParent, InputOutput, CopyFromParent,
      CWOverrideRedirect | CWBackPixel, &swa);

  XSelectInput(x->dpy, x->win, ExposureMask | KeyPressMask | StructureNotifyMask);

  x->gc = XCreateGC(x->dpy, x->win, 0, NULL);

  // Still handle WM_DELETE if WM is in the way (some setups ignore override_redirect)
  x->wm_delete = XInternAtom(x->dpy, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(x->dpy, x->win, &x->wm_delete, 1);

  XMapRaised(x->dpy, x->win);
  XFlush(x->dpy);

  // Ensure we actually receive key events even with override_redirect fullscreen
  XSetInputFocus(x->dpy, x->win, RevertToParent, CurrentTime);

  // Grab keyboard so Esc/q always works
int gk = XGrabKeyboard(x->dpy, x->win, True, GrabModeAsync, GrabModeAsync, CurrentTime);
if (gk != GrabSuccess) {
  fprintf(stderr, "Warning: XGrabKeyboard failed (%d). Esc/q may not work if window lacks focus.\n", gk);
}
XSync(x->dpy, False);

}

static void draw_centered_qr(XCtx *x, const QrCode *qr, int scale, int quiet, bool invert,
                            bool draw_text, const char *text_line) {
  const int qrSize = qr->size;
  const int total = qrSize + 2 * quiet;
  int pix = scale;

  int imgW = total * pix;
  int imgH = total * pix;

  int ox = (x->width - imgW) / 2;
  int oy = (x->height - imgH) / 2;

  // Clear background
  XSetForeground(x->dpy, x->gc, invert ? WhitePixel(x->dpy, x->screen) : BlackPixel(x->dpy, x->screen));
  XFillRectangle(x->dpy, x->win, x->gc, 0, 0, (unsigned)x->width, (unsigned)x->height);

  // Draw QR background (white)
  XSetForeground(x->dpy, x->gc, invert ? BlackPixel(x->dpy, x->screen) : WhitePixel(x->dpy, x->screen));
  XFillRectangle(x->dpy, x->win, x->gc, ox, oy, (unsigned)imgW, (unsigned)imgH);

  // Draw modules
  unsigned long fg = invert ? WhitePixel(x->dpy, x->screen) : BlackPixel(x->dpy, x->screen);
  XSetForeground(x->dpy, x->gc, fg);

  for (int y = 0; y < qrSize; y++) {
    for (int z = 0; z < qrSize; z++) {
      int v = qr->mods[idx(qrSize, z, y)] ? 1 : 0;
      if (!v) continue;
      int px0 = ox + (z + quiet) * pix;
      int py0 = oy + (y + quiet) * pix;
      XFillRectangle(x->dpy, x->win, x->gc, px0, py0, (unsigned)pix, (unsigned)pix);
    }
  }

if (draw_text && text_line) {
  int tx = 20;
  int ty = x->height - 30;
  unsigned long textcol = invert ? BlackPixel(x->dpy, x->screen)
                                 : WhitePixel(x->dpy, x->screen);
  XSetForeground(x->dpy, x->gc, textcol);
  XDrawString(x->dpy, x->win, x->gc, tx, ty, text_line, (int)strlen(text_line));
}

  XFlush(x->dpy);
}

/* ----------------------------- CLI ----------------------------- */

typedef struct {
  double hz;
  bool use_ms;
  enum QrEcc ecc;
  int scale;
  int quiet;
  bool invert;
  bool draw_text;
  int monitor;
  char payload_fmt[256];
  bool raw_only;
} Opts;

static enum QrEcc parse_ecc(const char *s) {
  if (!s || !*s) return QR_ECC_MEDIUM;
  if (strcasecmp(s, "L") == 0) return QR_ECC_LOW;
  if (strcasecmp(s, "M") == 0) return QR_ECC_MEDIUM;
  if (strcasecmp(s, "Q") == 0) return QR_ECC_QUARTILE;
  if (strcasecmp(s, "H") == 0) return QR_ECC_HIGH;
  die("Invalid ECC level: %s (use L/M/Q/H)", s);
  return QR_ECC_MEDIUM;
}

static void usage(const char *argv0) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "Options:\n"
    "  --hz <float>          Update rate (QR refresh) in Hz (default 60)\n"
    "  --ms | --sec          Encode unix time in milliseconds (default) or seconds\n"
    "  --ec <L|M|Q|H>         QR error correction (default M)\n"
    "  --scale <int>         Pixels per module (default 12)\n"
    "  --quiet <int>         Quiet zone modules (default 4)\n"
    "  --invert              Invert colors (black background, white modules)\n"
    "  --text                Draw a small text line with the payload\n"
    "  --monitor <int>       Monitor index (reserved; default 0)\n"
    "  --payload <fmt>       printf-style payload format (default: t_unix_ms=%llu)\n"
    "  --raw                 Payload is ONLY the timestamp number (ignores --payload)\n"
    "  --help                Show this help\n"
    "\n"
    "Controls:\n"
    "  Space: pause/resume\n"
    "  Esc/q: quit\n",
    argv0);
}

static Opts parse_args(int argc, char **argv) {
  Opts o;
  o.hz = 60.0;
  o.use_ms = true;
  o.ecc = QR_ECC_MEDIUM;
  o.scale = 12;
  o.quiet = 4;
  o.invert = false;
  o.draw_text = false;
  o.monitor = 0;
  o.raw_only = false;
  snprintf(o.payload_fmt, sizeof(o.payload_fmt), "t_unix_ms=%llu");

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (strcmp(a, "--help") == 0) {
      usage(argv[0]);
      exit(0);
    } else if (strcmp(a, "--hz") == 0 && i + 1 < argc) {
      o.hz = atof(argv[++i]);
      if (!(o.hz > 0.0 && isfinite(o.hz))) die("Invalid --hz value");
    } else if (strcmp(a, "--ms") == 0) {
      o.use_ms = true;
    } else if (strcmp(a, "--sec") == 0) {
      o.use_ms = false;
    } else if (strcmp(a, "--ec") == 0 && i + 1 < argc) {
      o.ecc = parse_ecc(argv[++i]);
    } else if (strcmp(a, "--scale") == 0 && i + 1 < argc) {
      o.scale = atoi(argv[++i]);
      if (o.scale <= 0) die("Invalid --scale");
    } else if (strcmp(a, "--quiet") == 0 && i + 1 < argc) {
      o.quiet = atoi(argv[++i]);
      if (o.quiet < 0) die("Invalid --quiet");
    } else if (strcmp(a, "--invert") == 0) {
      o.invert = true;
    } else if (strcmp(a, "--text") == 0) {
      o.draw_text = true;
    } else if (strcmp(a, "--monitor") == 0 && i + 1 < argc) {
      o.monitor = atoi(argv[++i]);
      if (o.monitor < 0) die("Invalid --monitor");
    } else if (strcmp(a, "--payload") == 0 && i + 1 < argc) {
      snprintf(o.payload_fmt, sizeof(o.payload_fmt), "%s", argv[++i]);
    } else if (strcmp(a, "--raw") == 0) {
      o.raw_only = true;
    } else {
      die("Unknown option: %s (try --help)", a);
    }
  }
  return o;
}

/* ----------------------------- Main ----------------------------- */

int main(int argc, char **argv) {
  Opts opt = parse_args(argc, argv);

  Display *dpy = XOpenDisplay(NULL);
  if (!dpy) die("XOpenDisplay failed. Is DISPLAY set?");

  XCtx x = {0};
  x.dpy = dpy;
  x11_fullscreen(&x, opt.monitor);

  // Work buffers (keep these reasonably sized; versions up to 10 => size 57 => 3249 modules)
  // We'll allocate generously.
  const size_t MODCAP = 6000; // > 57*57
  unsigned char *mods = (unsigned char *)calloc(MODCAP, 1);
  unsigned char *resv = (unsigned char *)calloc(MODCAP, 1);
  unsigned char *codewords = (unsigned char *)calloc(2048, 1);
  unsigned char *bitbuf = (unsigned char *)calloc(2048, 1);
  if (!mods || !resv || !codewords || !bitbuf) die("Out of memory");

  bool running = true;
  bool paused = false;
  uint64_t frame = 0;

  uint64_t interval_ns = (uint64_t)llround(1e9 / opt.hz);
  if (interval_ns == 0) interval_ns = 1;

  uint64_t next_ns = now_mono_ns();

  while (running) {
    // Handle events quickly
    while (XPending(x.dpy)) {
      XEvent ev;
      XNextEvent(x.dpy, &ev);
      if (ev.type == KeyPress) {
        KeySym ks = XLookupKeysym(&ev.xkey, 0);
        if (ks == XK_Escape || ks == XK_q) {
          running = false;
        } else if (ks == XK_space) {
          paused = !paused;
        }
      } else if (ev.type == ClientMessage) {
        if ((Atom)ev.xclient.data.l[0] == x.wm_delete) running = false;
      } else if (ev.type == ConfigureNotify) {
        x.width = ev.xconfigure.width;
        x.height = ev.xconfigure.height;
      }
    }

    if (!paused) {
      uint64_t t = opt.use_ms ? now_unix_ms() : now_unix_s();

      char payload[512];
      if (opt.raw_only) {
        snprintf(payload, sizeof(payload), "%" PRIu64, t);
      } else {
        // user fmt expects %llu for timestamp; we also offer %llu for frame by letting them include it:
        // easiest: if they want frame, they can embed it as text by putting e.g. "...&frame=%llu" but then
        // they'd need two args. We'll support exactly: fmt(timestamp, frame).
        // If their fmt uses only one, extra arg is ignored by snprintf? (No) -> we must always pass both
        // but format must match. We'll provide two helper replacements:
        // Use %llu for timestamp and %llu for frame if you want it.
        snprintf(payload, sizeof(payload), opt.payload_fmt,
                 (unsigned long long)t, (unsigned long long)frame);
      }

      // Generate QR
      QrCode qr = {0};
      int rc = make_qr_byte_mode((const unsigned char *)payload, (int)strlen(payload),
                                opt.ecc, &qr, mods, resv, codewords, bitbuf, MODCAP);
      if (rc != 0) {
        // If payload too long, fall back to raw timestamp
        snprintf(payload, sizeof(payload), "%" PRIu64, t);
        rc = make_qr_byte_mode((const unsigned char *)payload, (int)strlen(payload),
                               opt.ecc, &qr, mods, resv, codewords, bitbuf, MODCAP);
        if (rc != 0) die("QR generation failed (payload too long?)");
      }

      // Render centered
      draw_centered_qr(&x, &qr, opt.scale, opt.quiet, opt.invert, opt.draw_text, payload);

      frame++;
    }

    next_ns += interval_ns;
    sleep_until_ns(next_ns);
  }

  XDestroyWindow(x.dpy, x.win);
  XUngrabKeyboard(x.dpy, CurrentTime);
  XCloseDisplay(x.dpy);
  free(mods); free(resv); free(codewords); free(bitbuf);
  return 0;
}

