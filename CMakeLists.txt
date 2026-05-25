cmake_minimum_required(VERSION 3.18)
project(fast_sam_3dbody CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# CUDA  (optional – enables ONNX Runtime CUDA EP and ggml CUDA backend)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures" FORCE)
    enable_language(CUDA)
    set(GGML_CUDA     ON  CACHE BOOL "" FORCE)
    set(GGML_CUDA_F16 ON  CACHE BOOL "" FORCE)
    set(WITH_CUDA     ON)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(WITH_CUDA OFF)
    message(STATUS "CUDA not found – CPU-only build")
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
            # Download pre-built GPU package
            set(_ORT_VERSION "1.20.1")
            if(WITH_CUDA)
                set(_ORT_ARCHIVE "onnxruntime-linux-x64-gpu-${_ORT_VERSION}.tgz")
            else()
                set(_ORT_ARCHIVE "onnxruntime-linux-x64-${_ORT_VERSION}.tgz")
            endif()
            set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VERSION}/${_ORT_ARCHIVE}")
            set(_ORT_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime_dl")
            if(WITH_CUDA)
                set(_ORT_SUFFIX "-gpu")
            else()
                set(_ORT_SUFFIX "")
            endif()
            # After extraction + rename, contents live directly in _ORT_DOWNLOAD_DIR
            set(ONNX_RUNTIME_DIR "${_ORT_DOWNLOAD_DIR}")

            if(NOT EXISTS "${_ORT_DOWNLOAD_DIR}")
                message(STATUS "Downloading ONNX Runtime ${_ORT_VERSION} …")
                file(DOWNLOAD "${_ORT_URL}" "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}" SHOW_PROGRESS)
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/${_ORT_ARCHIVE}"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                )
                file(RENAME
                    "${CMAKE_BINARY_DIR}/onnxruntime-linux-x64${_ORT_SUFFIX}-${_ORT_VERSION}"
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
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui dnn)
message(STATUS "OpenCV ${OpenCV_VERSION}")

set(GRAPHICS_ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GraphicsEngine)

# ---------------------------------------------------------------------------
# AmMatrix  (built as subdirectory)
# ---------------------------------------------------------------------------
add_subdirectory(AmMatrix)

# ---------------------------------------------------------------------------
# fast_sam_3dbody  library
# ---------------------------------------------------------------------------
add_library(fast_sam_3dbody SHARED
    src/fast_sam_3dbody.cpp
    src/fast_sam_3dbody_capi.cpp
    src/bvh_writer.cpp
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_tri.c
    ${GRAPHICS_ENGINE_DIR}/ModelLoader/model_loader_transform_joints.c
)

target_include_directories(fast_sam_3dbody PUBLIC
    src
    ${GRAPHICS_ENGINE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/AmMatrix
    ${ggml_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fast_sam_3dbody PUBLIC
    ggml
    ${ORT_LIBS}
    ${OpenCV_LIBS}
    AmMatrix
    m
)

if(WITH_CUDA)
    target_compile_definitions(fast_sam_3dbody PUBLIC GGML_USE_CUDA FSB_CUDA)
    target_link_libraries(fast_sam_3dbody PUBLIC CUDA::cudart)
endif()

target_compile_options(fast_sam_3dbody PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_run  executable
# ---------------------------------------------------------------------------
add_executable(fast_sam_3dbody_run src/main.cpp)

target_link_libraries(fast_sam_3dbody_run PRIVATE fast_sam_3dbody)

target_compile_options(fast_sam_3dbody_run PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
)

# ---------------------------------------------------------------------------
# fast_sam_3dbody_render  executable  (OpenGL overlay renderer)
# ---------------------------------------------------------------------------
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
    rt
    m
    AmMatrix
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fast_sam_3dbody_render PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -g -O0>
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wno-unused-parameter -g -O0>
    )
else()
    target_compile_options(fast_sam_3dbody_render PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -O3 -march=native>
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wno-unused-parameter -O2>
    )
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== fast_sam_3dbody build ===")
message(STATUS "  CUDA support  : ${WITH_CUDA}")
message(STATUS "  ggml CUDA     : ${GGML_CUDA}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  ORT include   : ${ORT_INCLUDE_DIRS}")
message(STATUS "  Targets: fast_sam_3dbody (lib)  fast_sam_3dbody_run (exe)  fast_sam_3dbody_render (exe)")
