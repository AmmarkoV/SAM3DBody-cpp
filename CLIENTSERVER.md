# Networked client/server tracking — architecture

Goal: run the GPU-heavy SAM-3D-Body pipeline on a beefy server while a
**thin, underpowered PC** does nothing but grab webcam frames and ship them.
The client POSTs one JPEG per frame; the server runs inference and returns the
per-person MHR pose for that frame as the HTTP response body. The server also
keeps the authoritative `.bvh` files.

Transport is **AmmarServer** (HTTP server with C resource-handler callbacks)
on the server side and **AmmClient** (minimal HTTP client) on the client side,
both already vendored in `AmmarServer/`.

## Decisions locked

| Decision | Choice | Rationale |
|---|---|---|
| Regime | **Live, low-latency** | Always process the newest frame; lossy by design. |
| Split | **Thin client = camera only**; all YOLO/backbone/decoder/MHR/LBS on server | The weak PC must not decode/preprocess. |
| Transport | **HTTP POST per frame over AmmarServer / AmmClient**, keep-alive | Reuses the vendored libs; the HTTP response *is* the return channel — no custom framing. |
| Backpressure | **One frame in flight** (synchronous req/resp) + client drains camera to newest before each POST | Req/resp gives "single in-flight" for free; draining gives drop-latest. |
| Frame upload | **JPEG** (UVC MJPEG passthrough when available, else `cv::imencode` q≈80) | ~30–80 KB/frame; near-zero client CPU on MJPEG cams. |
| Response body | **Line-oriented clear text** (see Wire protocol) | Negligible size/parse vs the JPEG + 170 ms backbone; `curl`-debuggable; trivial multi-person tokenizing. |
| Where the BVH lives | **Server-side `BVHWriter`**, finalized on session end | We chose "poses back + BVH on server". |
| Identity | **Server-side bbox-IoU tracker → `write_frame_external`** | `FsbResult` has no id; same tracker feeds both the response `id=` and the `p_<id>.bvh` filenames so they agree. |
| GPU concurrency | **Single mutex around inference + writer** | One GPU; AmmarServer dispatches callbacks from a thread pool. |
| Packaging | **One binary `sam_3dbody_net`, `--server` / `--client` mode** | Same code is both ends — runs as either, even on the same machine; one thing to maintain. |
| Dependency | **AmmarServer is optional**: fetched by `tools/setup_ammarserver.sh`, a **symlink** in dev; net target is **skipped silently** if absent | Core build never depends on it; upstream fixes commit straight back through the symlink. |

## Data flow

```
CLIENT (thin)                              SERVER (GPU)
─────────────                              ────────────
cv::VideoCapture (BUFFERSIZE=1)            main(): fsb_load() once, BVHWriter::open() once,
  grab → drain to newest frame                     AmmServer_StartWithArgs(); register /infer,/close
  cv::imencode(".jpg", q≈80)  ── or ──
  UVC MJPEG passthrough (no re-encode)
        │  jpeg bytes
        ▼
  AmmClient_SendFile(inst,"/infer",
     "uploadedfile","f.jpg","image/jpeg",
     jpeg,jpegSize, keepAlive=1)
        │  multipart POST (jpeg + seq + t_ms)
        ╞═════════════ HTTP ════════════════►  /infer callback:
        │                                        jpeg = _FILES(rqst,"uploadedfile",VALUE,&n)
        │                                        seq  = _POSTuint(rqst,"seq")
        │                                      ── lock(g_mutex) ──
        │                                        bgr   = cv::imdecode(jpeg)
        │                                        n     = fsb_process_bgr(h,bgr,w,h,res,MAXP)
        │                                        ids   = tracker.assign(res)      // bbox-IoU
        │                                        bvh.write_frame_external(res,ids)// server keeps .bvh
        │                                        format_text(res,ids → rqst->content)
        │                                      ── unlock ──
        ◄═════════════ HTTP ════════════════   response body = SAM3D text (poses)
  AmmClient_Recv + AmmClient_seekEndOfHeader
  tokenize → overlay / consume
  (repeat)
        │  on exit:
        ╞═══► POST /close   → bvh.close(); reopen fresh for next session
```

Latency budget per frame ≈ JPEG encode (few ms) + uplink + `fsb_process_bgr`
(~170 ms TRT / ~270 ms CUDA) + downlink (~KB). The synchronous loop means the
client is busy-waiting on exactly one frame; every camera frame produced during
that wait is discarded by the drain step — that is the live/drop-latest behaviour.

## Wire protocol

### Request — `POST /infer` (multipart/form-data, keep-alive)

Built by `AmmClient_SendFile(...,"uploadedfile","frame.jpg","image/jpeg",...)`.
Optional form fields carried alongside the file (same mechanism the
`WebFramebuffer` demo uses for `width/height/framenumber`):

| Field | Meaning |
|---|---|
| `uploadedfile` | the JPEG bytes (the file) |
| `seq` | client frame counter, echoed back so the client can align poses to the frame it sent |
| `t_ms` | client monotonic capture time (ms); feeds the live filter `dt` and BVH frame-time |

Server handler registered with
`ENABLE_RECEIVING_FILES | DIFFERENT_PAGE_FOR_EACH_CLIENT`, and
`AmmServer_SetIntSettingValue(srv, AMMSET_MAX_POST_TRANSACTION_SIZE, 2*1024*1024)`
plus a resource-handler buffer ≥ 2 MB so a JPEG fits.

### Response — `text/plain`, line-oriented, one record-type token per line

The client dispatches on the **first whitespace token** of each line and
`strtod`-loops the rest. No nested parser. `Content-Length` delimits the body;
`END` is a debug convenience.

```
SAM3D 1 frame=1234 t_ms=170245 persons=2
P 0 id=0 bbox=120.0,80.0,320.0,470.0 focal=640.0
GROT -0.012 -1.571 0.034
CAMT 0.021 -0.104 4.503
BPOSE <133 floats>
HPOSE <108 floats>
BETAS shape=<45 floats> scale=<28 floats>          # only when changed for this id
KP2D 70 312.0,180.4 318.2,176.9 …                  # optional (only if !skip_body)
KP3D 70 …                                          # optional
P 1 id=3 bbox=… focal=…
GROT …
CAMT …
BPOSE …
HPOSE …
END frame=1234
```

Record types:

| Tag | Payload | Notes |
|---|---|---|
| `SAM3D` | `<version> frame= t_ms= persons=` | header; `persons=N` lets the client pre-size |
| `P` | `<index> id=<trackid> bbox=x1,y1,x2,y2 focal=` | opens a person block; `id` is the tracker id = `p_<id>.bvh` |
| `GROT` | 3 floats | `global_rot` Euler ZYX (radians) |
| `CAMT` | 3 floats | `pred_cam_t` = tx,ty,tz |
| `BPOSE` | 133 floats | `body_pose` |
| `HPOSE` | 108 floats | `hand_pose` (left 54 + right 54) |
| `BETAS` | `shape=… scale=…` | **sent only when changed** for that id (near-constant per identity) |
| `KP2D` / `KP3D` | `70 …` | optional; present only when the body model ran |
| `END` | `frame=` | terminator (debug) |

Format rules:
- **Precision `%.6g`** (≈7 sig digits; ~0.001° on Euler radians). The server
  keeps the authoritative `.bvh`, so the text stream is for consumption/overlay,
  not bit-exact round-trip.
- **Mandatory per person:** `P`, `GROT`, `CAMT`, `BPOSE`, `HPOSE`.
  **Optional:** `BETAS` (first frame of a track + on change), `KP2D`/`KP3D`.
- New tags may be appended in future; an unknown leading token is skipped, so
  older clients keep working (bump `SAM3D <version>` when the contract changes).

Sizing: ~251 mandatory floats/person ≈ 2.3 KB text; 5 persons ≈ 11 KB — well
under the JPEG uplink and microseconds to tokenize.

## One binary, two modes

A single executable `sam_3dbody_net` selected at runtime:

```bash
sam_3dbody_net --server --onnx-dir ./onnx --bvh ./out/p.bvh   # GPU box
sam_3dbody_net --client --host 192.168.1.50 --port 8080 --from 0   # webcam PC
```

Both modes share the SAM3D text serializer/parser, the option parsing, and the
JPEG helpers; only `--server` pulls in `fsb_*` + `BVHWriter` + `libAmmServer`,
and only `--client` pulls in `videoio` + `libAmmClient`. Keeping them in one
target means one build, one place for the protocol code, and the ability to run
both ends on the same machine for a loopback test.

## Server mode

Links `fsb::Pipeline` / the C API + OpenCV + `libAmmServer`.

State (process-global, one camera/client for v1):
- `FsbHandle pipeline` — `fsb_create` + `fsb_load` once in `main()` (model load is
  the slow part; never per request).
- `BVHWriter bvh` — `open(template, ...)` once; `set_frame_time` from observed
  cadence; fed via `write_frame_external`.
- `IoUTracker tracker` — small bbox-IoU greedy tracker (same logic as the live
  `BVHWriter` / offline Pass-2) producing stable ids.
- `std::mutex g_mutex` — guards `fsb_process_bgr` + tracker + writer.
- per-id `last_betas` cache to decide when to emit `BETAS`.

Endpoints (`AmmServer_AddResourceHandler`):
- `POST /infer` — the hot path above.
- `POST /close` — `lock; bvh.close(); bvh.open(...);` finalize this session's
  `.bvh` and arm a fresh one. Also auto-finalize on **idle timeout** (no `/infer`
  for N s) via the main loop.
- (optional) `GET /` — a one-line status/health page for `curl`.

Concurrency: AmmarServer may call the handler from multiple worker threads.
The single GPU + stateful tracker/writer ⇒ everything between `imdecode` and
filling `rqst->content` runs under `g_mutex`. With one client this is serial;
the mutex just makes it correct if a second connection appears.

Response is written by `memcpy`/`snprintf` into `rqst->content` (cap at
`rqst->MAXcontentSize`) and setting `rqst->contentSize` — exactly the pattern in
`WebFramebuffer/main.c`'s `framebuffer.jpg` handler.

## Client mode

Links OpenCV `videoio/imgcodecs` (+ `highgui` for overlay) + `libAmmClient`.

Loop:
1. `cap >> frame` with `cap.set(CAP_PROP_BUFFERSIZE, 1)`; if the driver still
   buffers, `grab()` in a tight loop to reach the newest frame, then `retrieve()`.
2. `cv::imencode(".jpg", frame, buf, {IMWRITE_JPEG_QUALITY, 80})` — or, for a
   native-MJPEG UVC cam, read the compressed frame and skip re-encoding.
3. `AmmClient_SendFile(inst, "/infer", "uploadedfile", "f.jpg", "image/jpeg",
   buf, bufSize, /*keepAlive=*/1)` with `seq`/`t_ms` as form fields.
4. `AmmClient_Recv(...)` → `AmmClient_seekEndOfHeader(...)` → tokenize the SAM3D
   text → draw overlay / hand off poses.
5. On exit: `AmmClient_SendFile(inst,"/close",...)` (empty), `AmmClient_Close`.

The client never queues frames; one in flight, newest only.

## Reuse map (unchanged existing code)

| Used as-is | From |
|---|---|
| `fsb_create / fsb_load / fsb_process_bgr`, `FsbConfig`, `FsbResult` | `src/fast_sam_3dbody_capi.h` |
| `BVHWriter::open / write_frame_external / close / set_frame_time` | `src/bvh_writer.{h,cpp}` |
| `AmmServer_StartWithArgs / AddResourceHandler / SetIntSettingValue`, `_FILES / _POST*`, `rqst->content` | `AmmarServer/src/AmmServerlib` |
| `AmmClient_Initialize / SendFile / Recv / seekEndOfHeader / Close` | `AmmarServer/src/AmmClient` |
| JPEG encode/decode | OpenCV `imgcodecs` |

New code is only: one `main()` with a `--server`/`--client` dispatch, the SAM3D
text serializer/parser (shared), and the small IoU tracker (or lift the one
already inside `BVHWriter`/offline).

## Dependency: AmmarServer (fetched, optional, symlinked in dev)

AmmarServer is **not** a hard dependency — the core build must succeed without
it, and its absence must not produce warnings.

- **`tools/setup_ammarserver.sh`** — mirrors `tools/setup_trt.sh`: if
  `./AmmarServer` is missing, `git clone https://github.com/AmmarkoV/AmmarServer`
  (or `git -C AmmarServer pull` if present), then build its libs
  (`libAmmServer`, `libAmmClient`). Idempotent; safe to re-run.
- **Dev convention** — `./AmmarServer` is a **symlink** to a working checkout of
  the library, so any fix made while building the net code is committed straight
  upstream. The setup script must not clobber an existing symlink (skip clone if
  `AmmarServer` already resolves to a real tree).
- **CMake detection** — probe for AmmarServer (e.g. the `AmmServerlib.h` /
  `AmmClient.h` headers + their libs under `./AmmarServer`, following the
  symlink). If found, add the `sam_3dbody_net` target and link against it; if
  **not** found, **skip the target with at most a single `message(STATUS …)`**
  ("AmmarServer not found — skipping sam_3dbody_net; run tools/setup_ammarserver.sh
  to enable"), never a `WARNING`/`SEND_ERROR`. The rest of the build is unaffected.

## Build / integration

- One CMake target `sam_3dbody_net`, added only when AmmarServer is detected.
- Links `libfast_sam_3dbody` + OpenCV + `libAmmServer` + `libAmmClient`
  (one binary needs both; the unused side is just dormant at runtime).
- Reference the AmmarServer libs from the detected/symlinked `./AmmarServer`
  tree (its own CMake/`make.sh` builds `libAmmServer*.a` / `libAmmClient.a`),
  rather than re-vendoring sources.

## Verification

1. **curl smoke test** — `curl -F uploadedfile=@assets/teaser.png -F seq=0 -F t_ms=0
   http://SERVER:8080/infer` prints a readable `SAM3D … P 0 …` block. Confirms the
   endpoint, multipart parse, and human-readability requirement in one shot.
2. **Loopback parity** — run client→server on localhost against a known clip;
   diff the server `.bvh` against `offline_sam_3dbody_render` on the same frames
   (expect close, modulo dropped frames + live-vs-offline smoothing).
3. **Multi-person** — a 2+ person clip yields one `P` block per person and stable
   `id=` across frames; `p_0.bvh`/`p_1.bvh` filenames match the streamed ids.
4. **Drop-latest under load** — throttle the link; verify end-to-end lag stays ≈
   one inference cycle (no unbounded growth) and the client always shows recent poses.

## Implementation status (validated 2026-06-07)

**Built and working end-to-end.**  `net/sam_3dbody_net.cpp` (one binary,
`--server`/`--client`), `tools/setup_ammarserver.sh`, and the optional
`sam_3dbody_net` CMake target are all in place.  Validated by streaming
`videos/boom.mp4` client→server on loopback: steady **~64 ms rtt (~15 Hz)**,
`persons=1` with a stable `id=0` tracked across frames, and a finalized
**367-frame** `*_0.bvh` after `/close`.  A `curl -F uploadedfile=@img.jpg
'http://host:port/infer'` returns the human-readable SAM3D text directly.

**AmmarServer fix (committed upstream via the `./AmmarServer` symlink).**
Multipart POSTs hung and 400'd because the receive-completion check
over-counted the expected size: `keepAnalyzingHTTPHeader()` tokenises the HTTP
header in place, overwriting its `\r\n\r\n`, so the later re-scan in
`HTTPRequestIsComplete()` locked onto the *first multipart part's* blank line
(346 B in) instead of the HTTP header end (204 B).  Since that 142-byte part
preamble is also inside `Content-Length`, the server waited forever for 142
bytes that never come (`recv` blocked the full `SO_RCVTIMEO`).  Fix in
`header_analysis/generic_header_tools.c`: capture `headerRAWHeadSize` **once on
the pristine buffer** at the top of `keepAnalyzingHTTPHeader()`, and only
fall back to re-scanning if that didn't happen (plus silenced the per-request
scanner debug `fprintf`s).

**Known limitation — `seq`/`t_ms`:** sent as URL query params, but AmmarServer
doesn't parse GET params on POST, so the server echoes `frame=0`.  Non-functional
here (the synchronous request/response means the client already knows which
frame it's awaiting; the BVH uses the nominal `--bvh-fps` frame time).  If
per-frame `t_ms` is ever needed server-side (e.g. to drive live-filter `dt`),
move them into the multipart form or parse the URI query manually.

## Out of scope / future

- **Multiple clients / cameras** — key all server state by `clientListID` (a map
  of `{tracker, BVHWriter, betas-cache}` per client) and put a single inference
  queue in front of the GPU instead of an inline mutex (seam toward a GPU-farm /
  broker model).
- **Record-then-process regime** — a lossless capture path that uploads every
  frame and runs the offline 5-pass server-side for best-quality BVH.
- **TLS / auth** — AmmarServer binds plain HTTP here; front with a reverse proxy
  if exposed beyond a trusted LAN.
- **WebSocket / WebRTC** — only if a browser/phone client or open-internet jitter
  control becomes a requirement.
```
