cmake_minimum_required(VERSION 3.18)
project(fast_sam_3dbody CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
if(WIN32)
    # TARGET_RUNTIME_DLLS is used to stage dependencies next to the binaries.
    if(CMAKE_VERSION VERSION_LESS 3.21)
        message(FATAL_ERROR "Windows builds require CMake 3.21 or newer")
    endif()
    set(MATH_LIB "")   # libm is folded into the CRT on Windows
    set(RT_LIB   "")   # librt is POSIX-only
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
    if(MSVC)
        add_compile_options(/utf-8 /wd4996)
    endif()
else()
    set(MATH_LIB m)
    set(RT_LIB   rt)
endif()

# RobustCapture (src/SAM3DBODY-cpp/v4l2_capture.cpp) uses raw V4L2 (linux/videodev2.h) to
# re-grab webcam frames the kernel flags as incomplete/corrupt, instead of
# handing back the torn image cv::VideoCapture's V4L2 backend would return.
# That header only exists on Linux, so probe for it explicitly rather than
# just relying on the outer NOT WIN32 guard (BSD/macOS are NOT WIN32 either
# but have no V4L2). When it's missing, FSB_HAVE_V4L2 is left undefined and
# src/SAM3DBODY-cpp/v4l2_capture.cpp compiles to a thin pass-through to cv::VideoCapture —
# same public API, old behavior.
include(CheckIncludeFileCXX)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    check_include_file_cxx("linux/videodev2.h" SAM3D_HAVE_V4L2_HEADER)
endif()
if(SAM3D_HAVE_V4L2_HEADER)
    message(STATUS "RobustCapture: raw V4L2 webcam capture enabled (FSB_HAVE_V4L2)")
else()
    message(STATUS "RobustCapture: linux/videodev2.h not found — falling back to plain cv::VideoCapture")
endif()

# ---------------------------------------------------------------------------
# Model files check
# ---------------------------------------------------------------------------
set(_ONNX_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/onnx")
set(_SENTINEL  "${_ONNX_DIR}/pipeline.gguf")   # representative file

# Opt-in only.  A configure step that silently starts a multi-GB download would
# break CI and packagers, so by default we just warn; the binaries themselves
# recover at runtime via ensure_models() -> tools/fetch_model.sh.
option(SAM3D_FETCH_MODELS "Download the model files at configure time" OFF)

if(NOT EXISTS "${_SENTINEL}")
    if(SAM3D_FETCH_MODELS)
        message(STATUS "SAM3D_FETCH_MODELS=ON — fetching models into ${_ONNX_DIR} …")
        if(WIN32)
            execute_process(
                COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass -File
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/fetch_model.ps1"
                    -Profile cuda -OnnxDir "${_ONNX_DIR}" -Yes
                RESULT_VARIABLE _FETCH_RC)
        else()
            execute_process(
                COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/fetch_model.sh"
                        shared cuda --onnx-dir "${_ONNX_DIR}" --yes
                RESULT_VARIABLE _FETCH_RC)
        endif()
        if(NOT _FETCH_RC EQUAL 0)
            message(WARNING "Model fetch failed (exit ${_FETCH_RC}) — "
                            "run 'bash tools/fetch_model.sh' by hand.")
        endif()
    elseif(WIN32)
        message(STATUS "Models absent: build and model-free tests are available. Download with "
            "powershell -File tools/fetch_model.ps1 -Profile cuda -Yes "
            "(use -Profile cpu for CPU inference). See knowledge/WINDOWS.md.")
    else()
        message(WARNING
            "\n"
            "  ┌─────────────────────────────────────────────────────────────────┐\n"
            "  │  Model files not found!                                         │\n"
            "  │                                                                 │\n"
            "  │  Fetch them with any of:                                        │\n"
            "  │    bash scripts/setup.sh                                        │\n"
            "  │    bash tools/fetch_model.sh shared cuda      (CUDA, ~5.2 GB)   │\n"
            "  │    bash tools/fetch_model.sh shared cpu       (CPU,  ~3.6 GB)   │\n"
            "  │    cmake -DSAM3D_FETCH_MODELS=ON ..                             │\n"
            "  │                                                                 │\n"
            "  │  The build succeeds either way — the binaries also fetch them   │\n"
            "  │  on first run when they are missing.                            │\n"
            "  └─────────────────────────────────────────────────────────────────┘\n"
        )
    endif()
else()
    message(STATUS "Model files found in ${_ONNX_DIR}")
endif()

# ---------------------------------------------------------------------------
# Backbone quantization  (build-time default; overridable at runtime via --backbone)
# ---------------------------------------------------------------------------
# After running tools/quantize_backbone.py you can switch the baked-in default
# filename from backbone.onnx to backbone_int8.onnx so all three binaries pick
# up the quantized model without needing the --backbone flag every invocation.
# The original backbone.onnx is never removed — both coexist in onnx/.
#
# Usage:  cmake -DSAM3D_BACKBONE_QUANT=ON ..
# ---------------------------------------------------------------------------
option(SAM3D_BACKBONE_QUANT
    "Default to backbone_int8.onnx instead of backbone.onnx (run tools/quantize_backbone.py first)"
    OFF)
if(SAM3D_BACKBONE_QUANT)
    message(STATUS "SAM3D_BACKBONE_QUANT=ON — default backbone: backbone_int8.onnx")
endif()

# ---------------------------------------------------------------------------
# SAM3D_TENSORRT — compile in the ONNX Runtime TensorRT execution provider.
#
# This only enables the code path; the EP is used at runtime when --trt is
# passed.  It needs no TensorRT SDK headers at build time (it uses ORT's own
# OrtTensorRTProviderOptions), but the NVIDIA TensorRT *runtime* libraries
# (libnvinfer.so, libnvonnxparser.so — matching the ORT build's TRT major
# version) must be on the loader path at run time, or --trt falls back to the
# CUDA EP.  Defaults ON for CUDA builds; harmless to leave on without TRT
# installed (--trt simply degrades to CUDA).
#
# Usage:  cmake -DSAM3D_TENSORRT=OFF ..   (to compile it out)
# ---------------------------------------------------------------------------
option(SAM3D_TENSORRT
    "Compile in the ONNX Runtime TensorRT EP (used at runtime via --trt)"
    ON)

# ---------------------------------------------------------------------------
# CUDA  (optional – enables ONNX Runtime CUDA EP and ggml CUDA backend)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)
# CUDAToolkit can report FOUND on just the runtime libs while nvcc is missing
# from PATH; enable_language(CUDA) would then hard-error with "No
# CMAKE_CUDA_COMPILER could be found". So feed it the nvcc that ships with the
# toolkit we found, then probe with check_language() and only enable CUDA if a
# working compiler is actually present — otherwise degrade to a CPU-only build.
include(CheckLanguage)
if(CUDAToolkit_FOUND AND NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
check_language(CUDA)
if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
    endif()
    enable_language(CUDA)
    set(GGML_CUDA     ON  CACHE BOOL "" FORCE)
    set(GGML_CUDA_F16 ON  CACHE BOOL "" FORCE)
    set(WITH_CUDA     ON)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(WITH_CUDA OFF)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA toolkit found but no nvcc compiler – CPU-only build "
                       "(put /usr/local/cuda/bin on PATH and re-run to enable CUDA)")
    else()
        message(STATUS "CUDA not found – CPU-only build")
    endif()
endif()

# Invoke in the directory that creates the target (including multiview).
function(sam3d_stage_runtime target)
    if(WIN32)
        file(GLOB _extra_dlls
            "${ONNX_RUNTIME_DIR}/lib/onnxruntime_providers*.dll"
            "${ONNX_RUNTIME_DIR}/bin/onnxruntime_providers*.dll"
            "${OpenCV_CONFIG_PATH}/../bin/opencv_videoio_ffmpeg*.dll"
            "${OpenCV_INSTALL_PATH}/bin/opencv_videoio_ffmpeg*.dll")
        if(SAM3D_GLEW_DLL AND target MATCHES "^(fast_sam_3dbody_render|test_wgl_context)$")
            list(APPEND _extra_dlls "${SAM3D_GLEW_DLL}")
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DSAM3D_RUNTIME_DLLS=$<TARGET_RUNTIME_DLLS:${target}>;${_extra_dlls}"
                "-DSAM3D_DESTINATION=$<TARGET_FILE_DIR:${target}>"
                -P "${PROJECT_SOURCE_DIR}/cmake/StageRuntime.cmake"
            VERBATIM)
    endif()
endfunction()

# ORT's prebuilt CUDA provider does not need nvcc at build time. This lets
# Windows users run GPU inference while compiling ggml and LBS on the CPU.
option(SAM3D_ONNX_CUDA "Use the GPU ONNX Runtime package (CUDA runtime required when running)" ${WITH_CUDA})

# ---------------------------------------------------------------------------
# ggml  (provides GGUF loading + small FFN inference for MHR/camera heads)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ggml)

# ---------------------------------------------------------------------------
# ONNX Runtime  (pre-built binaries from GitHub releases)
# ONNX_RUNTIME_DIR can be set via -DONNX_RUNTIME_DIR=/path/to/onnxruntime
# Otherwise we try to find it in common locations or download it.
# ---------------------------------------------------------------------------
if(NOT DEFINED ONNX_RUNTIME_DIR OR ONNX_RUNTIME_DIR STREQUAL "")
    # Try pkg-config / system install first
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "Found system ONNX Runtime via CMake config")
        set(ORT_LIBS onnxruntime::onnxruntime)
        set(ORT_INCLUDE_DIRS "")
    else()
        # Try common install paths
        foreach(_try
            "/usr/local/lib/cmake/onnxruntime"
            "/opt/onnxruntime"
            "$ENV{HOME}/.local"
        )
            if(EXISTS "${_try}/include/onnxruntime_cxx_api.h")
                set(ONNX_RUNTIME_DIR "${_try}")
                break()
            endif()
        endforeach()

        if(ONNX_RUNTIME_DIR STREQUAL "" OR NOT DEFINED ONNX_RUNTIME_DIR)
            # Download the pre-built package matching this platform/accelerator.
            set(_ORT_VERSION "1.20.1")
            if(SAM3D_ONNX_CUDA)
                set(_ORT_SUFFIX "-gpu")
            else()
                set(_ORT_SUFFIX "")
            endif()
            if(WIN32)
                set(_ORT_OS  "win-x64")
                set(_ORT_EXT "zip")
            else()
                set(_ORT_OS  "linux-x64")
                set(_ORT_EXT "tgz")
            endif()
            set(_ORT_DIRNAME "onnxruntime-${_ORT_OS}${_ORT_SUFFIX}-${_ORT_VERSION}")
            set(_ORT_ARCHIVE "${_ORT_DIRNAME}.${_ORT_EXT}")
            set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VERSION}/${_ORT_ARCHIVE}")
            # Keep CPU/GPU (and version/platform) packages separate so changing
            # SAM3D_ONNX_CUDA in an existing build never reuses the wrong runtime.
            set(_ORT_DOWNLOAD_ROOT "${CMAKE_BINARY_DIR}/onnxruntime_dl")
            set(_ORT_DOWNLOAD_DIR "${_ORT_DOWNLOAD_ROOT}/${_ORT_DIRNAME}")
            set(ONNX_RUNTIME_DIR "${_ORT_DOWNLOAD_DIR}")

            if(NOT EXISTS "${_ORT_DOWNLOAD_DIR}/include/onnxruntime_cxx_api.h")
                message(STATUS "Downloading ONNX Runtime ${_ORT_VERSION} (${_ORT_OS}${_ORT_SUFFIX}) …")
                file(DOWNLOAD "${_ORT_URL}" "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}"
                     SHOW_PROGRESS STATUS _ORT_DOWNLOAD_STATUS TLS_VERIFY ON)
                list(GET _ORT_DOWNLOAD_STATUS 0 _ORT_DOWNLOAD_RC)
                if(NOT _ORT_DOWNLOAD_RC EQUAL 0)
                    message(FATAL_ERROR "ONNX Runtime download failed: ${_ORT_DOWNLOAD_STATUS}. Set ONNX_RUNTIME_DIR to an extracted package.")
                endif()
                file(MAKE_DIRECTORY "${_ORT_DOWNLOAD_ROOT}")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}"
                    WORKING_DIRECTORY "${_ORT_DOWNLOAD_ROOT}"
                    RESULT_VARIABLE _ORT_EXTRACT_RC
                )
                if(NOT _ORT_EXTRACT_RC EQUAL 0)
                    message(FATAL_ERROR "Failed to extract ${_ORT_ARCHIVE}")
                endif()
            endif()
        endif()
    endif()
endif()

if(NOT TARGET onnxruntime::onnxruntime)
    set(ORT_INCLUDE_DIRS "${ONNX_RUNTIME_DIR}/include")
    set(ORT_LIB_DIR      "${ONNX_RUNTIME_DIR}/lib")
    # These are derived from ONNX_RUNTIME_DIR, not independent user settings.
    # CMake caches find_* results across configures; discard them before looking
    # in a newly selected CPU/GPU package or an explicit replacement directory.
    unset(ORT_LIB CACHE)
    unset(ORT_LIB)
    unset(ORT_DLL CACHE)
    unset(ORT_DLL)
    find_library(ORT_LIB onnxruntime PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}")
    if(WIN32)
        find_file(ORT_DLL onnxruntime.dll PATHS "${ORT_LIB_DIR}" "${ONNX_RUNTIME_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ORT_LIB}" IMPORTED_LOCATION "${ORT_DLL}")
    else()
        set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_LIB}")
    endif()
    set(ORT_LIBS onnxruntime::onnxruntime)
    message(STATUS "ONNX Runtime at ${ONNX_RUNTIME_DIR}")
endif()

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui dnn video)
message(STATUS "OpenCV ${OpenCV_VERSION}")

set(GRAPHICS_ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/GraphicsEngine)

# ---------------------------------------------------------------------------
# AmMatrix
# ---------------------------------------------------------------------------
# AmMatrix ships its own CMakeLists.txt that hardcodes GCC-only flags
# (-march=native, -s) and links libm/librt, so it cannot configure under MSVC.
# Its sources are, however, portable C math (no GL/X11/OpenCV includes). On
# Windows we therefore compile them straight into the core library and skip the
# subdirectory; on Linux we keep the existing static-library build unchanged.
if(WIN32)
    set(AMMATRIX_SOURCES
        src/AmMatrix/quaternions.c
        src/AmMatrix/matrix3x3Tools.c
        src/AmMatrix/matrix4x4Tools.c
        src/AmMatrix/matrixCalculations.c
        src/AmMatrix/matrixMultiplicationOptimization.c
        src/AmMatrix/matrixOpenGL.c
        src/AmMatrix/simpleRenderer.c
        src/AmMatrix/ocvStaging.c
        src/AmMatrix/collisions.c
        src/AmMatrix/solids.c
        src/AmMatrix/solveHomography.c
        src/AmMatrix/solveLinearSystemGJ.c
    )
    set(AMMATRIX_LIB "")
else()
    add_subdirectory(src/AmMatrix)
    set(AMMATRIX_SOURCES "")
    set(AMMATRIX_LIB AmMatrix)
endif()

# ---------------------------------------------------------------------------
# fast_sam_3dbody  library
# ---------------------------------------------------------------------------
set(MOCAP_DIR ${GRAPHICS_ENGINE_DIR}/MotionCaptureLoader)
set(TRAJ_DIR  ${GRAPHICS_ENGINE_DIR}/TrajectoryParser)

add_library(fast_sam_3dbody SHARED
    src/SAM3DBODY-cpp/fast_sam_3dbody.cpp
    src/SAM3DBODY-cpp/fast_sam_3dbody_capi.cpp
    src/SAM3DBODY-cpp/bvh_writer.cpp
    src/SAM3DBODY-cpp/bvh_shm.cpp
    src/SAM3DBODY-cpp/offline_passes.cpp
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_tri.c
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_transform_joints.c
    # MotionCaptureLoader (BVH I/O) ------------------------------------------------
    ${MOCAP_DIR}/bvh_loader.c
    ${MOCAP_DIR}/calculate/bvh_project.c
    ${MOCAP_DIR}/calculate/bvh_transform.c
    ${MOCAP_DIR}/calculate/bvh_to_tri_pose.c
    ${MOCAP_DIR}/calculate/smoothing.c
    ${MOCAP_DIR}/export/bvh_to_bvh.c
    ${MOCAP_DIR}/edit/bvh_cut_paste.c
    ${MOCAP_DIR}/edit/bvh_rename.c
    ${MOCAP_DIR}/import/fromBVH.c
    ${TRAJ_DIR}/InputParser_C.c
    ${AMMATRIX_SOURCES}   # empty on Linux (linked as a library instead)
)

target_include_directories(fast_sam_3dbody PUBLIC
    src/SAM3DBODY-cpp
    ${GRAPHICS_ENGINE_DIR}
    ${MOCAP_DIR}
    ${TRAJ_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/AmMatrix
    ${ggml_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fast_sam_3dbody PUBLIC
    ggml
    ${ORT_LIBS}
    ${OpenCV_LIBS}
    ${AMMATRIX_LIB}
    ${MATH_LIB}
)

if(WIN32)
    # M_PI and friends are not defined by default in the MSVC <cmath>.
    target_compile_definitions(fast_sam_3dbody PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(fast_sam_3dbody PRIVATE FSB_BUILD_DLL)
    set_target_properties(fast_sam_3dbody PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(WITH_CUDA)
    target_sources(fast_sam_3dbody PRIVATE src/SAM3DBODY-cpp/mhr_lbs_cuda.cu)
    target_compile_definitions(fast_sam_3dbody PUBLIC GGML_USE_CUDA FSB_CUDA)
    target_link_libraries(fast_sam_3dbody PUBLIC CUDA::cudart)
endif()

if(SAM3D_BACKBONE_QUANT)
    target_compile_definitions(fast_sam_3dbody PUBLIC SAM3D_BACKBONE_QUANT)
endif()

# TensorRT EP: only meaningful for CUDA builds (it dlopens the NVIDIA TRT runtime
# at session-create time and degrades to the CUDA EP if it's not present).
if(SAM3D_TENSORRT AND SAM3D_ONNX_CUDA)
    target_compile_definitions(fast_sam_3dbody PRIVATE USE_TENSORRT_EP)
    message(STATUS "SAM3D_TENSORRT=ON — TensorRT EP compiled in (enable at runtime with --trt)")
endif()

# ---------------------------------------------------------------------------
# SharedMemoryVideoBuffers  (live webcam→robot zero-copy transport, Linux only)
# ---------------------------------------------------------------------------
# tools/setup_gmr.sh clones this into ${SHM_DIR} and builds its .so for the
# Python consumer.  Here we compile its single C source straight into our core
# library (POSIX shm/rt — Linux only) and define FSB_SHM so main.cpp's --bvh-shm
# publisher is active.  Absent (setup_gmr.sh not run / non-Linux) → FSB_SHM off,
# and the webcam pipeline transparently uses the stdout/tempfile path instead.
set(_SHM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/SharedMemoryVideoBuffers)
set(_SHM_SRC ${_SHM_DIR}/src/c/sharedMemoryVideoBuffers.c)
if(UNIX AND NOT APPLE AND EXISTS "${_SHM_SRC}")
    target_sources(fast_sam_3dbody PRIVATE ${_SHM_SRC})
    target_include_directories(fast_sam_3dbody PUBLIC ${_SHM_DIR}/src/c)
    target_compile_definitions(fast_sam_3dbody PUBLIC FSB_SHM)
    target_link_libraries(fast_sam_3dbody PUBLIC ${RT_LIB} pthread)
    message(STATUS "SharedMemoryVideoBuffers found — live shm transport compiled in (FSB_SHM)")
else()
    message(STATUS "SharedMemoryVideoBuffers not found — webcam→robot uses the stdout/tempfile fallback "
                   "(run tools/setup_gmr.sh to enable the faster shm transport)")
endif()

target_compile_options(fast_sam_3dbody PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O3 -march=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:/O2>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_run  executable
# ---------------------------------------------------------------------------
add_executable(fast_sam_3dbody_run src/SAM3DBODY-cpp/main.cpp)

target_link_libraries(fast_sam_3dbody_run PRIVATE fast_sam_3dbody)

target_compile_options(fast_sam_3dbody_run PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O3 -march=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:/O2>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_render  executable  (OpenGL overlay renderer)
#
# Linux uses GLX; Windows uses the matching WGL context implementation.
# ---------------------------------------------------------------------------
option(SAM3D_BUILD_RENDERER "Build the live OpenGL viewer (GLX on Linux, WGL on Windows)" ON)
if(SAM3D_BUILD_RENDERER)
    find_package(OpenGL REQUIRED)
    find_package(GLEW   REQUIRED)

    # FindGLEW's module-mode target is UNKNOWN IMPORTED on Windows and its
    # IMPORTED_LOCATION is the .lib, so TARGET_RUNTIME_DLLS cannot see the DLL.
    if(WIN32)
        if(GLEW_USE_STATIC_LIBS)
            set_property(TARGET GLEW::GLEW APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS GLEW_STATIC)
        else()
            get_target_property(_glew_type GLEW::GLEW TYPE)
            if(_glew_type STREQUAL "UNKNOWN_LIBRARY")
                find_file(SAM3D_GLEW_DLL NAMES glew32.dll GLEW.dll
                    HINTS "${GLEW_ROOT}" "$ENV{GLEW_ROOT}" "${GLEW_INCLUDE_DIR}/.."
                    PATH_SUFFIXES bin/Release/x64 bin REQUIRED)
            endif()
        endif()
    endif()

    if(WIN32)
        set(SAM3D_GL_CONTEXT ${GRAPHICS_ENGINE_DIR}/System/wgl3.c)
        set(SAM3D_WINDOW_LIBS user32 gdi32)
    else()
        set(SAM3D_GL_CONTEXT ${GRAPHICS_ENGINE_DIR}/System/glx3.c)
        set(SAM3D_WINDOW_LIBS X11)
    endif()

    add_executable(fast_sam_3dbody_render
        src/render/fast_sam_3dbody_render.cpp
        src/SAM3DBODY-cpp/v4l2_capture.cpp
        ${SAM3D_GL_CONTEXT}
        ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_tri.c
        ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_transform_joints.c
    )

    target_compile_definitions(fast_sam_3dbody_render PRIVATE USE_GLEW)
    if(SAM3D_HAVE_V4L2_HEADER)
        target_compile_definitions(fast_sam_3dbody_render PRIVATE FSB_HAVE_V4L2)
    endif()

    target_include_directories(fast_sam_3dbody_render PRIVATE
        src/SAM3DBODY-cpp
        ${GRAPHICS_ENGINE_DIR}
        ${ORT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
    )

    target_link_libraries(fast_sam_3dbody_render PRIVATE
        fast_sam_3dbody
        OpenGL::GL
        GLEW::GLEW
        ${OpenCV_LIBS}
        ${SAM3D_WINDOW_LIBS}
        ${RT_LIB}
        ${MATH_LIB}
        ${AMMATRIX_LIB}
    )

    if(MSVC)
        target_compile_options(fast_sam_3dbody_render PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(fast_sam_3dbody_render PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -g -O0>
            $<$<COMPILE_LANGUAGE:C>:-Wall -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -g -O0>
        )
    else()
        target_compile_options(fast_sam_3dbody_render PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O3 -march=native>
            $<$<COMPILE_LANGUAGE:C>:-Wall -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O2>
        )
    endif()
else()
    message(STATUS "Live OpenGL viewer disabled (SAM3D_BUILD_RENDERER=OFF)")
endif()

# ---------------------------------------------------------------------------
# offline_sam_3dbody_render  executable
#
# Multi-pass offline BVH extractor.  Same fsb::Pipeline + BVHWriter as the
# live binaries; no OpenGL dependency (this tool never opens a window).
# Only the per-pass orchestration code is unique to this target.
# ---------------------------------------------------------------------------
add_executable(offline_sam_3dbody_render
    src/render/offline_sam_3dbody_render.cpp
)

target_include_directories(offline_sam_3dbody_render PRIVATE
    src/SAM3DBODY-cpp
    ${GRAPHICS_ENGINE_DIR}
    ${ORT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(offline_sam_3dbody_render PRIVATE
    fast_sam_3dbody
    ${OpenCV_LIBS}
    ${MATH_LIB}
)

target_compile_options(offline_sam_3dbody_render PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O3 -march=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:/O2>
)

# ---------------------------------------------------------------------------
# sam_3dbody_net  executable  (networked client/server — CLIENTSERVER.md)
#
# One binary, two modes (--server / --client).  Optional: depends on AmmarServer
# (HTTP server lib) + AmmClient (HTTP client lib), fetched/built by
# tools/setup_ammarserver.sh.  ./AmmarServer is typically a symlink to a dev
# checkout.  When the libs are absent the target is silently skipped — a single
# STATUS line, never a warning — so the core build is unaffected.  Linked against
# the shared libs (the static AmmarServer lib pulls in sibling libs the .so
# bundles), with an rpath so they resolve without LD_LIBRARY_PATH.
# ---------------------------------------------------------------------------
set(_AMM_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/AmmarServer")
set(_AMM_SRV_HDR "${_AMM_DIR}/src/AmmServerlib/AmmServerlib.h")
set(_AMM_CLI_HDR "${_AMM_DIR}/src/AmmClient/AmmClient.h")
set(_AMM_SRV_LIB "${_AMM_DIR}/src/AmmServerlib/libAmmarServer.so")
set(_AMM_CLI_LIB "${_AMM_DIR}/src/AmmClient/libAmmClient.so")

if(NOT WIN32 AND EXISTS "${_AMM_SRV_HDR}" AND EXISTS "${_AMM_CLI_HDR}"
   AND EXISTS "${_AMM_SRV_LIB}" AND EXISTS "${_AMM_CLI_LIB}")
    add_executable(sam_3dbody_net src/net/sam_3dbody_net.cpp)
    target_include_directories(sam_3dbody_net PRIVATE
        src/SAM3DBODY-cpp
        ${OpenCV_INCLUDE_DIRS}
        "${_AMM_DIR}/src/AmmServerlib"
        "${_AMM_DIR}/src/AmmClient"
    )
    target_link_libraries(sam_3dbody_net PRIVATE
        fast_sam_3dbody
        ${OpenCV_LIBS}
        "${_AMM_SRV_LIB}"
        "${_AMM_CLI_LIB}"
        pthread
        ${RT_LIB}
        ${MATH_LIB}
    )
    set_target_properties(sam_3dbody_net PROPERTIES
        BUILD_RPATH   "${_AMM_DIR}/src/AmmServerlib;${_AMM_DIR}/src/AmmClient"
        INSTALL_RPATH "${_AMM_DIR}/src/AmmServerlib;${_AMM_DIR}/src/AmmClient"
    )
    target_compile_options(sam_3dbody_net PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O3 -march=native>
    )
    message(STATUS "AmmarServer found — building sam_3dbody_net (networked client/server)")
else()
    message(STATUS "AmmarServer not found — skipping sam_3dbody_net; run tools/setup_ammarserver.sh to enable")
endif()

# ---------------------------------------------------------------------------
# QR timecode synchronization tool (optional; needs X11/Xrandr/qrencode)
# ---------------------------------------------------------------------------
if(NOT WIN32)
    add_subdirectory(src/synchronization)
endif()

# ---------------------------------------------------------------------------
# multi-view capture & fusion (MULTIVIEW_PLAN.md) — .calib loader so far
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(src/multiview)

foreach(_target fast_sam_3dbody_run offline_sam_3dbody_render fast_sam_3dbody_render)
    if(TARGET ${_target})
        add_test(NAME ${_target}_help COMMAND ${_target}
            --onnx-dir definitely-missing-models --from 999 --help)
        set_tests_properties(${_target}_help PROPERTIES TIMEOUT 15
            ENVIRONMENT "SAM3D_AUTO_FETCH=0"
            PASS_REGULAR_EXPRESSION "Usage:"
            FAIL_REGULAR_EXPRESSION "Loading backbone|models missing|Failed to load pipeline")
    endif()
endforeach()

# These regression tests need no model files or network access.
add_executable(test_python_abi_layout tests/test_python_abi_layout.c)
target_include_directories(test_python_abi_layout PRIVATE src/SAM3DBODY-cpp)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME python_abi COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_abi.py"
        --probe $<TARGET_FILE:test_python_abi_layout>
        --lib-dir $<TARGET_FILE_DIR:fast_sam_3dbody>)
endif()
add_executable(test_portable_getline tests/test_portable_getline.c)
add_test(NAME portable_getline COMMAND test_portable_getline)
if(WIN32)
    # CTest launched by PowerShell 7 otherwise passes its incompatible module
    # search path to Windows PowerShell 5.1. Let that child derive its own path.
    add_test(NAME powershell_model_manifest COMMAND ${CMAKE_COMMAND} -E env
        --unset=PSModulePath powershell.exe -NoProfile
        -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fetch_model.ps1")
    set_tests_properties(powershell_model_manifest PROPERTIES TIMEOUT 30)
    add_executable(windows_worker_pool_test src/SAM3DBODY-cpp/windows_worker_pool_test.cpp)
    add_test(NAME windows_worker_pool COMMAND windows_worker_pool_test)
    if(SAM3D_BUILD_RENDERER)
        add_executable(test_wgl_context tests/test_wgl_context.c ${SAM3D_GL_CONTEXT})
        target_link_libraries(test_wgl_context PRIVATE OpenGL::GL GLEW::GLEW user32 gdi32)
        option(SAM3D_TEST_OPENGL "Run the WGL test (requires an interactive desktop and OpenGL 3.3 driver)" OFF)
        if(SAM3D_TEST_OPENGL)
            add_test(NAME wgl_context COMMAND test_wgl_context)
            set_tests_properties(wgl_context PROPERTIES TIMEOUT 30)
        endif()
    endif()

    # ORT providers and OpenCV's FFmpeg backend are loaded dynamically, so
    # TARGET_RUNTIME_DLLS alone cannot discover them.
    foreach(_target fast_sam_3dbody fast_sam_3dbody_run offline_sam_3dbody_render
            fast_sam_3dbody_render test_wgl_context)
        if(TARGET ${_target})
            sam3d_stage_runtime(${_target})
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if(SAM3D_BUILD_RENDERER)
    set(_VIS_STATUS "enabled (fast_sam_3dbody_render)")
    set(_TARGETS_STATUS "fast_sam_3dbody (lib)  fast_sam_3dbody_run (exe)  fast_sam_3dbody_render (exe)  offline_sam_3dbody_render (exe)")
else()
    set(_VIS_STATUS "disabled")
    set(_TARGETS_STATUS "fast_sam_3dbody (lib)  fast_sam_3dbody_run (exe)  offline_sam_3dbody_render (exe)")
endif()

message(STATUS "")
message(STATUS "=== fast_sam_3dbody build ===")
message(STATUS "  Platform      : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  CUDA support  : ${WITH_CUDA}")
message(STATUS "  ORT GPU pkg   : ${SAM3D_ONNX_CUDA}")
message(STATUS "  ggml CUDA     : ${GGML_CUDA}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  ORT include   : ${ORT_INCLUDE_DIRS}")
message(STATUS "  Visualization : ${_VIS_STATUS}")
message(STATUS "  Targets: ${_TARGETS_STATUS}")
