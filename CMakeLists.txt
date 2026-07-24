cmake_minimum_required(VERSION 3.18)
project(fast_sam_3dbody CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
# Windows is a HEADLESS-ONLY target: the live OpenGL/X11 overlay renderer
# (fast_sam_3dbody_render) is not built there. The CLI (fast_sam_3dbody_run)
# and the offline BVH extractor (offline_sam_3dbody_render) are. See the
# configure-time notice further down where the renderer would be defined.
if(WIN32)
    set(MATH_LIB "")   # libm is folded into the CRT on Windows
    set(RT_LIB   "")   # librt is POSIX-only
else()
    set(MATH_LIB m)
    set(RT_LIB   rt)
endif()

# ---------------------------------------------------------------------------
# Model files check
# ---------------------------------------------------------------------------
set(_ONNX_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/onnx")
set(_ONNX_ZIP  "${CMAKE_CURRENT_SOURCE_DIR}/SAM3DBody-cpp-onnx-models.zip")
set(_SENTINEL  "${_ONNX_DIR}/pipeline.gguf")   # representative file

if(NOT EXISTS "${_SENTINEL}")
    if(EXISTS "${_ONNX_ZIP}")
        message(STATUS "")
        message(STATUS "  *** Model zip found: ${_ONNX_ZIP}")
        message(STATUS "  *** Extract it before running: unzip SAM3DBody-cpp-onnx-models.zip")
        message(STATUS "")
    else()
        message(WARNING
            "\n"
            "  ┌─────────────────────────────────────────────────────────────────┐\n"
            "  │  Model files not found!                                         │\n"
            "  │                                                                 │\n"
            "  │  Download SAM3DBody-cpp-onnx-models.zip from HuggingFace:       │\n"
            "  │  https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models      │\n"
            "  │                                                                 │\n"
            "  │  Then extract it at the repo root:                              │\n"
            "  │    unzip SAM3DBody-cpp-onnx-models.zip                          │\n"
            "  │                                                                 │\n"
            "  │  The build will succeed but the runtime will fail without them. │\n"
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
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures" FORCE)
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
            if(WITH_CUDA)
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
            set(_ORT_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime_dl")
            # After extraction + rename, contents live directly in _ORT_DOWNLOAD_DIR
            set(ONNX_RUNTIME_DIR "${_ORT_DOWNLOAD_DIR}")

            if(NOT EXISTS "${_ORT_DOWNLOAD_DIR}")
                message(STATUS "Downloading ONNX Runtime ${_ORT_VERSION} (${_ORT_OS}${_ORT_SUFFIX}) …")
                file(DOWNLOAD "${_ORT_URL}" "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}" SHOW_PROGRESS)
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                )
                file(RENAME
                    "${CMAKE_BINARY_DIR}/${_ORT_DIRNAME}"
                    "${_ORT_DOWNLOAD_DIR}"
                )
            endif()
        endif()
    endif()
endif()

if(NOT TARGET onnxruntime::onnxruntime)
    set(ORT_INCLUDE_DIRS "${ONNX_RUNTIME_DIR}/include")
    set(ORT_LIB_DIR      "${ONNX_RUNTIME_DIR}/lib")
    find_library(ORT_LIB onnxruntime PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
    )
    set(ORT_LIBS onnxruntime::onnxruntime)
    message(STATUS "ONNX Runtime at ${ONNX_RUNTIME_DIR}")
endif()

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui dnn video)
message(STATUS "OpenCV ${OpenCV_VERSION}")

set(GRAPHICS_ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GraphicsEngine)

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
        AmMatrix/quaternions.c
        AmMatrix/matrix3x3Tools.c
        AmMatrix/matrix4x4Tools.c
        AmMatrix/matrixCalculations.c
        AmMatrix/matrixMultiplicationOptimization.c
        AmMatrix/matrixOpenGL.c
        AmMatrix/simpleRenderer.c
        AmMatrix/ocvStaging.c
        AmMatrix/collisions.c
        AmMatrix/solids.c
        AmMatrix/solveHomography.c
        AmMatrix/solveLinearSystemGJ.c
    )
    set(AMMATRIX_LIB "")
else()
    add_subdirectory(AmMatrix)
    set(AMMATRIX_SOURCES "")
    set(AMMATRIX_LIB AmMatrix)
endif()

# ---------------------------------------------------------------------------
# fast_sam_3dbody  library
# ---------------------------------------------------------------------------
set(MOCAP_DIR ${GRAPHICS_ENGINE_DIR}/MotionCaptureLoader)
set(TRAJ_DIR  ${GRAPHICS_ENGINE_DIR}/TrajectoryParser)

add_library(fast_sam_3dbody SHARED
    src/fast_sam_3dbody.cpp
    src/fast_sam_3dbody_capi.cpp
    src/bvh_writer.cpp
    src/bvh_shm.cpp
    src/offline_passes.cpp
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
    src
    ${GRAPHICS_ENGINE_DIR}
    ${MOCAP_DIR}
    ${TRAJ_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/AmMatrix
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
endif()

if(WITH_CUDA)
    target_sources(fast_sam_3dbody PRIVATE src/mhr_lbs_cuda.cu)
    target_compile_definitions(fast_sam_3dbody PUBLIC GGML_USE_CUDA FSB_CUDA)
    target_link_libraries(fast_sam_3dbody PUBLIC CUDA::cudart)
endif()

if(SAM3D_BACKBONE_QUANT)
    target_compile_definitions(fast_sam_3dbody PUBLIC SAM3D_BACKBONE_QUANT)
endif()

# TensorRT EP: only meaningful for CUDA builds (it dlopens the NVIDIA TRT runtime
# at session-create time and degrades to the CUDA EP if it's not present).
if(SAM3D_TENSORRT AND WITH_CUDA)
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
set(_SHM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SharedMemoryVideoBuffers)
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
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_run  executable
# ---------------------------------------------------------------------------
add_executable(fast_sam_3dbody_run src/main.cpp)

target_link_libraries(fast_sam_3dbody_run PRIVATE fast_sam_3dbody)

target_compile_options(fast_sam_3dbody_run PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_render  executable  (OpenGL overlay renderer)
#
# Linux-only: it is built on a GLX/X11 OpenGL context (glx3.c) with no Windows
# (WGL) equivalent in-tree. Windows builds are headless — the renderer is
# skipped and a notice is printed instead.
# ---------------------------------------------------------------------------
if(NOT WIN32)
    find_package(OpenGL REQUIRED)
    find_package(GLEW   REQUIRED)

    add_executable(fast_sam_3dbody_render
        render/fast_sam_3dbody_render.cpp
        ${GRAPHICS_ENGINE_DIR}/System/glx3.c
        ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_tri.c
        ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_transform_joints.c
    )

    target_compile_definitions(fast_sam_3dbody_render PRIVATE USE_GLEW)

    target_include_directories(fast_sam_3dbody_render PRIVATE
        src
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
        X11
        ${RT_LIB}
        ${MATH_LIB}
        AmMatrix
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
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
    message(STATUS "")
    message(STATUS "  ┌─────────────────────────────────────────────────────────────────┐")
    message(STATUS "  │  Windows build: HEADLESS ONLY                                   │")
    message(STATUS "  │                                                                 │")
    message(STATUS "  │  Live visualization is NOT supported on Windows. The OpenGL     │")
    message(STATUS "  │  overlay renderer (fast_sam_3dbody_render) needs GLX/X11 and    │")
    message(STATUS "  │  will not be built.                                             │")
    message(STATUS "  │                                                                 │")
    message(STATUS "  │  Use these instead:                                             │")
    message(STATUS "  │    fast_sam_3dbody_run        – CLI pose/keypoint output        │")
    message(STATUS "  │    offline_sam_3dbody_render  – offline BVH extraction          │")
    message(STATUS "  └─────────────────────────────────────────────────────────────────┘")
    message(STATUS "")
endif()

# ---------------------------------------------------------------------------
# offline_sam_3dbody_render  executable
#
# Multi-pass offline BVH extractor.  Same fsb::Pipeline + BVHWriter as the
# live binaries; no OpenGL dependency (this tool never opens a window).
# Only the per-pass orchestration code is unique to this target.
# ---------------------------------------------------------------------------
add_executable(offline_sam_3dbody_render
    render/offline_sam_3dbody_render.cpp
)

target_include_directories(offline_sam_3dbody_render PRIVATE
    src
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
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
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
    add_executable(sam_3dbody_net net/sam_3dbody_net.cpp)
    target_include_directories(sam_3dbody_net PRIVATE
        src
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
add_subdirectory(synchronization)

# ---------------------------------------------------------------------------
# multi-view capture & fusion (MULTIVIEW_PLAN.md) — .calib loader so far
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(multiview)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if(WIN32)
    set(_VIS_STATUS "DISABLED (headless build — Windows not supported)")
    set(_TARGETS_STATUS "fast_sam_3dbody (lib)  fast_sam_3dbody_run (exe)  offline_sam_3dbody_render (exe)")
else()
    set(_VIS_STATUS "enabled (fast_sam_3dbody_render)")
    set(_TARGETS_STATUS "fast_sam_3dbody (lib)  fast_sam_3dbody_run (exe)  fast_sam_3dbody_render (exe)  offline_sam_3dbody_render (exe)")
endif()

message(STATUS "")
message(STATUS "=== fast_sam_3dbody build ===")
message(STATUS "  Platform      : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  CUDA support  : ${WITH_CUDA}")
message(STATUS "  ggml CUDA     : ${GGML_CUDA}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  ORT include   : ${ORT_INCLUDE_DIRS}")
message(STATUS "  Visualization : ${_VIS_STATUS}")
message(STATUS "  Targets: ${_TARGETS_STATUS}")
