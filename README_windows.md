I have created a comprehensive `README_windows.md` that covers the prerequisites, build process, and usage for the Windows port of `SAM3DBody-cpp`.

### README_windows.md

```markdown
# SAM3DBody-cpp: Windows Build & Usage Guide

This document provides instructions for building and running **SAM3DBody-cpp** on Windows using Visual Studio and CMake.

## 1. Prerequisites

Before building, ensure you have the following installed:

*   **Visual Studio 2022** (with "Desktop development with C++" workload).
*   **CMake** (version 3.18 or higher).
*   **CUDA Toolkit** (Optional, recommended for GPU acceleration).
*   **OpenCV**: Download and extract the Windows build (e.g., v4.x).
*   **GLEW**: Required for the OpenGL renderer.
    *   The project expects GLEW at `C:\MoCap\glew\glew-2.3.1`.
    *   If installed elsewhere, update the paths in `CMakeLists.txt` (lines 379-380).

## 2. Dependency Setup

### Environment Variables
Set the following environment variables to help CMake find your dependencies:

*   `OpenCV_DIR`: Path to your OpenCV build directory (e.g., `C:\opencv\build`).
*   Add OpenCV and CUDA `bin` directories to your system `PATH` to ensure DLLs are found at runtime.

### Model Files
Download the models from [HuggingFace](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models) and extract them into an `onnx/` folder at the repository root.

## 3. Building the Project

Open **PowerShell** in the repository root and run:

```powershell
# Create and enter build directory
mkdir build
cd build

# Configure the project for Visual Studio 2022 x64
# Note: Adjust paths to ONNX_RUNTIME_DIR and OpenCV_DIR to match your installation
cmake .. -G "Visual Studio 17 2022" -A x64 `
    "-DONNX_RUNTIME_DIR=C:\MoCap\onnx_rt\onnxruntime-win-x64-gpu-1.26.0" `
    "-DOpenCV_DIR=C:\MoCap\opencv-4.10.0\opencv\build"

# Build in Release mode
cmake --build . --config Release --parallel
```

The executables (`fast_sam_3dbody_run.exe`, `fast_sam_3dbody_render.exe`, etc.) will be located in `build\Release\`.

## 4. Running the Pipeline

Use the provided PowerShell scripts to handle paths and model flags automatically.

### Live Visualization (with 3D Overlay)
```powershell
.\scripts\video.ps1 --from path\to\your_video.mp4
```
*   **Controls**: Use the mouse to rotate/zoom the 3D view. Press `ESC` to exit.
*   **Saving**: Add `--save output.mp4` to render the visualization to a video file.

### Headless BVH Extraction
```powershell
.\scripts\offline_video.ps1 --from path\to\your_video.mp4 --bvh output_name
```
*   This generates `output_name_0.bvh`, `output_name_1.bvh`, etc., for each tracked person.

## 6. Troubleshooting

*   **Missing DLLs**: If the program fails to start (e.g., error `0xc000007b`), ensure `opencv_world*.dll`, `onnxruntime.dll`, and `glew32.dll` are in your `PATH` or copied to the same folder as the `.exe`.
*   **CUDA Errors**: If you have an incompatible GPU or missing drivers, use `--cuda -1` to force CPU inference (note: this will be very slow).
*   **OpenGL Errors**: Ensure your GPU drivers support OpenGL 3.3 or higher.
```
