project( AmMatrix )
cmake_minimum_required(VERSION 3.5...4.3)

#Make fast, lean and platform independent binaries..
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
    set(CMAKE_C_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
endif()

OPTION(INTEL_OPTIMIZATIONS OFF)

if (INTEL_OPTIMIZATIONS)
add_definitions(-DINTEL_OPTIMIZATIONS)
endif(INTEL_OPTIMIZATIONS)

if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

add_library(
            AmMatrix STATIC
            collisions.c
            quaternions.c
            matrix3x3Tools.c
            matrix4x4Tools.c
            matrixCalculations.c
            matrixCalculations.h
            matrixMultiplicationOptimization.c
            matrixMultiplicationOptimization.h
            matrixOpenGL.c
            matrixOpenGL.h
            ocvStaging.c
            ocvStaging.h
            solveHomography.c
            solveLinearSystemGJ.c
            simpleRenderer.c
            solids.c
           )
if(NOT WIN32)
    target_link_libraries(AmMatrix m rt )
endif()

add_subdirectory(TestCPUOptimizedInstructionSet/)

# ---------------------------------------------------------------------------
# Remove directory overrides to let CMake manage output paths automatically.
# ---------------------------------------------------------------------------
# set_target_properties(AmMatrix PROPERTIES
#                        DEBUG_POSTFIX "D"
#                        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                       )
