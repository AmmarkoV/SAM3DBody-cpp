# QR Timecode Synchronization

Displays fullscreen QR codes encoding live Unix timestamps. Point one or more cameras at the screen while recording — the embedded timecodes let you align footage from different sources to a common clock after the fact.

## How It Works

1. Run one of the display tools on a monitor visible to all cameras.
2. Record your footage normally. Each frame that captures the QR code carries an absolute timestamp.
3. Use the parent repo's `tracker.py` to extract QR payloads from the recordings. The CSV output includes `qr_*_t_unix_ms`, `qr_*_frame`, and `qr_*_hz` columns for post-hoc alignment.

## Files

| File | Description |
|---|---|
| `xQRSync.c` | Standalone X11 display — no external QR library, embedded generator |
| `libQRSync.c` | X11 display using `libqrencode` — adds dynamic Hz oscillation and multi-screen tiling |
| `sync.html` | Browser-based display — zero install, works on any platform |
| `qrcode.js` | QR library used by `sync.html` |
| `makeqrcode.sh` | Generates 60 static QR PNG frames (one per second) using `qrencode` CLI |
| `Makefile` | Builds `libQRSync.c` → `xqr_time_sync` |

## Build

### `xQRSync.c` (no external QR library)

```bash
gcc -O2 -std=c11 -Wall -Wextra -pedantic xQRSync.c -o xqr_time_sync -lX11 -lXrandr -lm
```

### `libQRSync.c` (requires libqrencode)

```bash
sudo apt install libqrencode-dev
make
```

## Usage

### C programs (`xQRSync.c` / `libQRSync.c`)

```bash
./xqr_time_sync [options]
```

**Controls:** `Esc` / `q` to quit, `Space` to pause/resume.

**Common options:**

| Option | Default | Description |
|---|---|---|
| `--hz <float>` | `60` | QR update rate in Hz |
| `--ms` / `--sec` | `--ms` | Timestamp unit: milliseconds or seconds |
| `--ec <L\|M\|Q\|H>` | `M` | QR error correction level |
| `--scale <int>` | `12` | Pixels per QR module |
| `--quiet <int>` | `4` | Quiet zone border in modules |
| `--invert` | off | White modules on black background |
| `--text` | off | Render payload text on screen |
| `--payload <fmt>` | `t_unix_ms=%llu` | printf-style format; `%llu` = timestamp, second `%llu` = frame count |
| `--raw` | off | Payload is the raw timestamp number only |

**`libQRSync.c`-only options:**

| Option | Description |
|---|---|
| `--dhz <min> <max> <T>` | Oscillate Hz between min and max, stepping every T seconds |
| `--tile <int>` | Tile QR across N horizontal screen segments |
| `--setX <int>` / `--setY <int>` | Override position within each tile segment |

**Examples:**

```bash
# 30 Hz, millisecond timestamps, show payload text
./xqr_time_sync --ms --hz 30 --text

# Custom payload matching tracker.py's QR parser
./xqr_time_sync --ms --hz 30 --payload "t_perf_ms=%llu&t_unix_ms=%llu&frame=%llu&hz=30"

# Oscillate between 25–60 Hz, changing every 5 seconds
./xqr_time_sync --dhz 25 60 5
```

### Browser (`sync.html`)

Open `sync.html` in a browser and press `F11` for fullscreen. Controls on the page let you adjust Hz and zoom.

Default payload: `perf=<tPerf>&t=<tUnix>&f=<frame>&hz=<rate>`

### Static frames (`makeqrcode.sh`)

```bash
./makeqrcode.sh
# Output: frames/image_000000.png … frames/image_000059.png
```

Each PNG encodes `ts_unix=<epoch>&ts_iso=<ISO8601>&frame=<index>`.

## Dependencies

| Tool | Required by |
|---|---|
| `libx11-dev`, `libxrandr-dev` | Both C programs |
| `libqrencode-dev` | `libQRSync.c` only |
| Any modern browser | `sync.html` |
| `qrencode` CLI | `makeqrcode.sh` |
