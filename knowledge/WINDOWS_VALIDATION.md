# Windows validation record

The Windows branch was tested locally on 2026-09-09, based on upstream commit
`a4190de3b31e396e4dc273085282da4fbceb3adf`. This records build and runtime smoke
tests, not a pose-accuracy evaluation or a performance benchmark.

## Environment

| Component | Tested version |
|---|---|
| OS | Windows 11 x64, build 26200 |
| Compiler | Visual Studio 2022 Build Tools, MSVC 14.44.35207 |
| SDK / CMake | Windows SDK 10.0.26100 / CMake 4.3.4 |
| Shell / Python | Windows PowerShell 5.1 / Python 3.11 x64 |
| OpenCV / GLEW | Official OpenCV 4.10.0 vc16 x64 / GLEW 2.2.0 |
| ONNX Runtime | Official 1.20.1 Windows x64 GPU package |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |
| NVIDIA driver | 610.62 |
| CUDA runtime packages | cuBLAS 12.4.5.8, cuDNN 9.1.0.70, CUDA runtime 12.4.127, cuFFT 11.2.1.3 |
| ggml checkout | `7840aaba1989c6deeefede1d77d5aaf8f52b947e` |

There was no nvcc toolchain installed for the inference tests. Native ggml/LBS
ran on CPU; ONNX Runtime used CUDA. The NVIDIA runtime DLL directories were
added to the process PATH. This distinguishes the runtime dependency from the
optional native CUDA build toolchain.

## Build and regression checks

The PowerShell build helper completed a native Release build, including the
DLL, CLI, WGL viewer, offline extractor and available multiview targets.
CTest passed all 12 enabled tests with `-TestOpenGL`: CLI/viewer/offline help,
C/Python ABI, portable getline, PowerShell model manifest, Windows worker pool,
WGL context/readback, calibration, extrinsics, and the two synchronization tests.
The model downloader test includes 17 offline checks.

Targeted reconfiguration checks verified CPU/GPU/explicit ORT package-path
selection in one build directory without retaining stale library cache values.
Those selection checks used directory junctions to an existing package; they
do not constitute a separate CPU runtime test. Generated MSVC Debug projects
use `/Od /RTC1`, and the CLI's Debug compilation step passed; a complete Debug
link and inference run was not performed.

The ABI check compares every field offset, field size and structure size
against a compiled C probe. It also loads the actual DLL and checks handle
creation/destruction and invalid-input guards. The WGL test compiles shaders,
draws and reads back pixels, exercises hidden and visible windows, resize,
close/Escape handling and context recreation.

The Windows GitHub Actions workflow passed actionlint 1.7.12 locally. The local
checks alone do not establish that a hosted workflow has passed; consult the
checks attached to the actual GitHub commit for that result.

## Actual inference and rendering

The PowerShell downloader fetched and verified all nine files in the `trt`
profile (about 1.80 GiB). These FP16 model files were executed on the **CUDA
execution provider**, without TensorRT, using explicit model filenames:

```powershell
$env:SAM3D_AUTO_FETCH = '0'
.\build\windows\Release\fast_sam_3dbody_run.exe `
  --onnx-dir .\onnx --cuda 0 `
  --backbone backbone_fp16_trt.onnx --decoder decoder_fp16.onnx `
  --from .\build\dancing.jpg --headless --max-persons 1 `
  --out .\build\dance.csv --bvh .\build\dance.bvh

.\build\windows\Release\fast_sam_3dbody_render.exe `
  --onnx-dir .\onnx --cuda 0 `
  --backbone backbone_fp16_trt.onnx --decoder decoder_fp16.onnx `
  --from .\build\dancing.jpg --headless --max-persons 1 --frames 1 `
  --save-frames .\build\dance-render
```

The image run returned one person with finite 3D keypoints. The CSV contained
212 columns, including 70 three-dimensional points. The native WGL viewer
saved a 2250 x 1500 JPEG with the reconstructed body mesh over the source image;
the overlay was inspected visually. Refined pose was not used in this test.

A 16-frame excerpt (zero-based source frames 200 through 215) of OpenCV's
`vtest.avi` was written at 10 FPS and processed by the offline extractor:

```powershell
.\build\windows\Release\offline_sam_3dbody_render.exe `
  --onnx-dir .\onnx --cuda 0 `
  --backbone backbone_fp16_trt.onnx --decoder decoder_fp16.onnx `
  --from .\build\vtest-busy.avi --max-persons 3 --thresh 0.25 `
  --bvh .\build\offline.bvh --static-scene --min-track-frames 1 `
  --no-refined-pose
```

It completed tracking and zero-phase smoothing, exporting two BVH tracks with
16 and 10 frames. Each motion row had all 498 declared channels; every value
was finite and the declared frame counts matched the data. The clip contains
more visible pedestrians than the detector returned; this test establishes
pipeline operation, not detection recall or reconstruction accuracy.

A separate actual-DLL ctypes smoke test used source frame 200: capacity three
returned two people with distinct bounding boxes and finite, nonzero keypoints
and 127-joint skeletons. Capacity one returned one person. Sentinel values
following both result arrays remained intact. For this test only, the FP16
graphs were hard-linked into a temporary model directory under the C API's
default filenames, with the original external-data sidecar filenames retained.
The public Python CLI still uses the default model names described in
[WINDOWS.md](WINDOWS.md); this does not add a Python FP16-selection option.

ONNX Runtime JSON profiles from a separate 16-frame video run contained CUDA
execution events for all three sessions: backbone, decoder and detector.
Some nodes also ran on CPU. CUDA use was checked from provider assignments,
not inferred only from a command-line flag.

## Inputs and limits

The input assets were used locally and are not added to this repository:

- [SAM 3D Body dancing sample](https://github.com/facebookresearch/sam-3d-body/blob/main/notebook/images/dancing.jpg),
  SHA256 `0112b0a32ea5860db6a6fc700804751528341a02bf07fed0329a0c2610f905d3`.
- [OpenCV vtest.avi sample](https://github.com/opencv/opencv/blob/5.x/samples/data/vtest.avi),
  SHA256 `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.

No inference FPS guarantee is derived from these short, varying-person-count
clips. Camera backends, TensorRT, native nvcc compilation, refined models,
Blender import, exhaustive non-ASCII path support, Linux runtime regression
and standalone installer packaging were not validated by these tests. Model
weights and CUDA/cuDNN runtime binaries are not included in the source change.
