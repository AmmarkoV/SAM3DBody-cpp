# Native Windows build

The Windows targets include the C++ inference CLI, the offline BVH extractor,
the shared DLL for Python/C callers, and a Win32/WGL OpenGL viewer. Bash and WSL
are not required for the native build. The Linux setup remains available in
[WSL.md](WSL.md).

Run the commands below in PowerShell **from the repository root**:

```powershell
Set-Location D:\AI\SAM3DBody-cpp
```

The build stages dependent runtime DLLs next to the executables, but it does not
create a standalone redistributable folder. The renderer still uses repository
resources such as `src/render/default.vert`, `src/render/default.frag`,
`onnx/body_mesh.tri`, and BVH templates under `bvh/`. Keep the checkout and model
files available; copying only `build/windows/Release` elsewhere is insufficient.

## Prerequisites

Use x64 components throughout:

- Visual Studio 2022 or Build Tools 2022 with **Desktop development with C++**,
  the MSVC x64 toolset and a Windows SDK.
- CMake **3.21 or newer**, plus Git, on `PATH`.
- Windows PowerShell **5.1 or newer** and `curl.exe` for the model downloader.
- The official [OpenCV 4.10.0 Windows package](https://github.com/opencv/opencv/releases/tag/4.10.0).
  Extract `opencv-4.10.0-windows.exe`; the examples assume the resulting
  `opencv/build` directory is `D:\deps\opencv\build`. Its x64 `vc16` libraries
  are used with the Visual Studio 2022 generator. Point `-OpenCVDir` at the
  extracted `build` directory containing `OpenCVConfig.cmake`.
- For the viewer, the [GLEW 2.2.0 Windows binaries](https://github.com/nigels-com/glew/releases/tag/glew-2.2.0).
  The example root `D:\deps\glew-2.2.0` should contain `include/GL/glew.h`,
  `lib/Release/x64/glew32.lib`, and `bin/Release/x64/glew32.dll`.
  A GPU display driver supporting OpenGL **3.3** is required to run the viewer.

CMake fetches ggml and, unless an existing ONNX Runtime is supplied, downloads
the appropriate ONNX Runtime **1.20.1** Windows x64 package. These dependency
downloads are separate from the model downloads. The helper reuses the fetched
ggml checkout on subsequent builds; pass
`-CMakeArgs '-DFETCHCONTENT_UPDATES_DISCONNECTED=OFF'` to request dependency updates.

## Build

For an NVIDIA GPU and the WGL viewer:

```powershell
.\scripts\build_windows.ps1 `
  -OpenCVDir D:\deps\opencv\build `
  -GLEWRoot D:\deps\glew-2.2.0 `
  -Gpu
```

The default Visual Studio generator targets x64 and builds Release into
`build/windows/Release`. The script runs CTest after building. A separate CPU
build without the OpenGL viewer needs no GLEW:

```powershell
.\scripts\build_windows.ps1 `
  -OpenCVDir D:\deps\opencv\build `
  -BuildDir .\build\windows-cpu `
  -Headless
```

`-Gpu` selects the prebuilt GPU ONNX Runtime package and enables its CUDA
execution provider. **It does not require nvcc to compile this project.** When
nvcc is unavailable, native ggml/LBS code runs on the CPU while ONNX inference
can use the GPU. If CMake finds a working CUDA compiler and toolkit, it can
also build the native CUDA paths.

| Script option | Effect |
|---|---|
| `-Gpu` | Set `SAM3D_ONNX_CUDA=ON`; select GPU ORT when CMake downloads ORT. |
| `-Headless` | Disable the viewer target with `SAM3D_BUILD_RENDERER=OFF`; retain the CLI, DLL and offline extractor. |
| `-TestOpenGL` | Include the model-free WGL test in CTest; requires the viewer build, an interactive desktop session and an OpenGL 3.3 driver. |
| `-OnnxRuntimeDir D:\deps\onnxruntime-win-x64-gpu-1.20.1` | Use an extracted ORT package containing `include` and `lib`; when using `-Gpu`, supply the GPU package. |
| `-CudaArchitectures 89` | Forward `CMAKE_CUDA_ARCHITECTURES` for native CUDA compilation when nvcc is available; it does not change the prebuilt ORT package. |
| `-BuildDir .\build\windows` | Choose the CMake build directory. |
| `-Configuration Release` | Select Release, RelWithDebInfo or Debug. Use Release for these examples. |
| `-Jobs 4` | Limit parallel build jobs. |
| `-SkipTests` | Build without running CTest. |

Automatically downloaded CPU and GPU ORT packages use separate, versioned
directories under `build/windows/onnxruntime_dl`, so switching `-Gpu` selects
the corresponding package. An explicit `-OnnxRuntimeDir` takes precedence:
when switching modes, update that argument to match the intended package.

If script execution is blocked, invoke the script with a process-local policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\build_windows.ps1 `
  -OpenCVDir D:\deps\opencv\build `
  -GLEWRoot D:\deps\glew-2.2.0 -Gpu
```

## GPU runtime DLLs

The GPU ORT 1.20.1 package still needs compatible **CUDA 12.x and cuDNN 9.x
runtime DLLs**, plus an NVIDIA driver, when the program runs. The project does
not bundle those NVIDIA runtime dependencies. See the
[ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

Add the directories actually containing your CUDA/cuDNN DLLs to the current
PowerShell session before running a native executable or Python. For example,
if those are the directories used by your installation:

```powershell
$env:PATH = "D:\deps\cuda12\bin;D:\deps\cudnn9\bin;$env:PATH"
```

`-OnnxRuntimeDir` selects ORT's own package; it does not install CUDA, cuDNN or
TensorRT. Inspect provider initialization messages to confirm the desired
provider loaded. A missing DLL error can refer to a dependency of
`onnxruntime_providers_cuda.dll`, even when that provider DLL itself is present.

## Download models

The PowerShell downloader reads the same manifest as `tools/fetch_model.sh`.
Use `-List` to inspect the selection without downloading or creating directories:

```powershell
.\tools\fetch_model.ps1 -Profile cuda -List
.\tools\fetch_model.ps1 -Profile cuda -OnnxDir .\onnx -Yes
```

| Profile | Files selected, in addition to shared files |
|---|---|
| `cpu` | `backbone_fp32.onnx` and its sidecar, plus `decoder_fp16.onnx` and its sidecar. |
| `cuda` | The stock `backbone.onnx`/sidecar and `decoder.onnx`, intended for CUDA inference. |
| `trt` | `backbone_fp16_trt.onnx`/sidecar and `decoder_fp16.onnx`/sidecar. These are model files; selecting the profile does not install or activate TensorRT. |
| `refined` | Additional iterative decoder graphs and `pipeline_refined.gguf`; explicitly opt in alongside a base profile. |
| `libreyolo` | The optional LibreYOLO detector. |
| `all` | The CPU, CUDA and TRT base profiles, with duplicate files removed. Refined and LibreYOLO remain opt-in. |

`shared` is implied by every profile. Multiple profiles can be combined, for
example `-Profile cuda,refined`. `-Revision` overrides the Hugging Face revision;
the default is `SAM3D_HF_REVISION` if set, otherwise `main`. The manifest size
and SHA256 are still enforced when choosing a different revision.

Existing files are skipped only after size and SHA256 verification. Downloads
go to `.partial` files and replace the final name only after verification.
Incomplete files can resume; `-Force` starts a fresh download. A corrupt file
is reported as invalid and is never silently accepted. `SAM3D_AUTO_FETCH=0`
blocks downloads even with `-Yes`; `SAM3D_AUTO_FETCH=1` skips the prompt.

## Run the native executables

The examples use your own input file at `D:\data\person.jpg` or
`D:\data\clip.mp4`; replace those paths with existing files. Continue to run
from the repository root so shader and BVH template paths resolve.

For the stock CUDA profile and a GPU ORT build:

```powershell
.\build\windows\Release\fast_sam_3dbody_run.exe `
  --onnx-dir .\onnx --gguf .\onnx\pipeline.gguf `
  --yolo .\onnx\yolo.onnx --cuda 0 --from D:\data\person.jpg
```

For a CPU-only run, first fetch `-Profile cpu`, then use the CPU build and
`--cuda -1`:

```powershell
.\build\windows-cpu\Release\fast_sam_3dbody_run.exe `
  --onnx-dir .\onnx --gguf .\onnx\pipeline.gguf `
  --yolo .\onnx\yolo.onnx --cuda -1 --from D:\data\person.jpg
```

To use the smaller FP16 model files with the **CUDA execution provider**,
download `-Profile trt` and explicitly select both model filenames:

```powershell
.\tools\fetch_model.ps1 -Profile trt -Yes
.\build\windows\Release\fast_sam_3dbody_run.exe `
  --onnx-dir .\onnx --gguf .\onnx\pipeline.gguf `
  --yolo .\onnx\yolo.onnx --cuda 0 `
  --backbone backbone_fp16_trt.onnx --decoder decoder_fp16.onnx `
  --from D:\data\person.jpg
```

This command does not need TensorRT runtime libraries. Only add `--trt` after
installing TensorRT runtime DLLs compatible with your ORT build and making them
discoverable. Model precision and execution provider are separate choices.

The viewer and offline extractor accept the same model-selection flags. With
the FP16 files above, a viewer invocation is:

```powershell
.\build\windows\Release\fast_sam_3dbody_render.exe `
  --onnx-dir .\onnx --gguf .\onnx\pipeline.gguf `
  --yolo .\onnx\yolo.onnx --cuda 0 `
  --backbone backbone_fp16_trt.onnx --decoder decoder_fp16.onnx `
  --from D:\data\clip.mp4
```

For offline BVH output, substitute `offline_sam_3dbody_render.exe` and add
`--bvh D:\data\capture.bvh`. The offline extractor enables refined pose by
default, so also fetch `-Profile refined`, or add `--no-refined-pose` to run
with only the base models downloaded above. Windows reports missing models
with manual download instructions instead of launching the Bash downloader.
The build-time `-Headless` switch omits the viewer;
the viewer's separate runtime `--headless` option uses a hidden WGL window and
still requires OpenGL. The offline extractor does not need a GL context.

## Python frontend and ABI

Use a 64-bit Python interpreter with NumPy and `opencv-python` for the
lightweight frontend. The 3D/PyTorch and ROS frontends retain their additional
dependencies. For the stock **cuda** profile:

```powershell
.\tools\fetch_model.ps1 -Profile cuda -Yes
python .\python\fast_sam_3dbody_frontend.py `
  --lib-dir .\build\windows --cuda 0 --from D:\data\person.jpg
```

`--lib-dir build/windows` also searches its `Release`, `RelWithDebInfo`, `Debug`
and `MinSizeRel` subdirectories. The shared `python/fsb_ctypes.py` binding loads
the cdecl DLL, registers dependency directories, and checks the native ABI
version and both structure sizes before binding inference functions. A stale
DLL or mismatched Python checkout fails with a clear ABI error; rebuild and use
matching files instead of bypassing the check.

The C/Python API uses the default model filenames. The explicit
`--backbone`/`--decoder` overrides in this guide belong to the native CLI;
the current Python frontend does not expose those model-selection options.

## Tests and current scope

See [WINDOWS_VALIDATION.md](WINDOWS_VALIDATION.md) for the tested environment,
actual GPU inference, WGL rendering and BVH export results.

Run the compiled regression suite separately with:

```powershell
ctest --test-dir .\build\windows -C Release --output-on-failure
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_fetch_model.ps1
```

CTest includes the C/Python layout and DLL lifecycle checks when a Python
interpreter is found. `-TestOpenGL` additionally enables the WGL context,
shader/pixel-readback, resize and teardown test. These tests do not require
model downloads; they do not establish camera compatibility, pose accuracy or
inference performance.

Windows support currently has these boundaries:

- The POSIX network server/client target, shared-memory transport, raw V4L2
  capture and X11 QR timecode display remain outside the native Windows build.
  Windows capture uses OpenCV's capture path.
- ArUco-dependent multiview tools are optional. When the `aruco` module is
  absent from the selected OpenCV package, CMake skips those tools; supplying
  an appropriate OpenCV contrib build enables their build paths. This is
  separate from the Linux-only X11 QR display utility.
- The documented build paths do not imply validation of every camera backend,
  native nvcc configuration, TensorRT runtime combination, or all non-ASCII
  model/media paths. Keep those checks separate from basic native build tests.
- Shell wrappers for FFmpeg, GMR or ROS workflows may need their own Windows
  adaptations and additional dependencies. Use the native executables for the
  commands in this guide.

## Attribution

The WGL implementation was adapted from **beemsoft's Windows port** in
[PR #13](https://github.com/AmmarkoV/SAM3DBody-cpp/pull/13), with attribution
retained in `src/GraphicsEngine/System/wgl3.c`. The current implementation adds
the OpenGL 3.3 context setup and the current window callback/title/cleanup
integration. The repository's existing license remains applicable.
