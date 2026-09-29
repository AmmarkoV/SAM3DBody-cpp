# h1_bridge — safety-supervised bridge from tools/gmr_stream.py to a Unitree H1-2.
#
#   cmake -S robot/h1_bridge -B robot/h1_bridge/build && cmake --build robot/h1_bridge/build
#   robot/h1_bridge/build/test_bridge          # no robot / SDK needed
#
# With OpenCV (highgui) present, h1_bridge also gets --ui / --manual.
#
# The safety core + tests build with no dependencies.  The h1_bridge executable
# additionally needs unitree_sdk2 (default: a sibling checkout ../unitree_sdk2 of
# this repo; override with -DUNITREE_SDK2_DIR=...).
cmake_minimum_required(VERSION 3.10)
project(h1_bridge CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_library(h1_bridge_core STATIC kinematics.cpp manual.cpp safety.cpp shm_reader.cpp)
target_include_directories(h1_bridge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(h1_bridge_core PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_executable(test_bridge test_bridge.cpp)
target_link_libraries(test_bridge h1_bridge_core)
target_compile_options(test_bridge PRIVATE -Wall -Wextra)
add_test(NAME test_bridge COMMAND test_bridge)

set(UNITREE_SDK2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../unitree_sdk2" CACHE PATH "unitree_sdk2 checkout")
if (EXISTS "${UNITREE_SDK2_DIR}/CMakeLists.txt")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${UNITREE_SDK2_DIR} unitree_sdk2 EXCLUDE_FROM_ALL)
    add_executable(h1_bridge main.cpp)
    target_link_libraries(h1_bridge h1_bridge_core unitree_sdk2)
    target_compile_options(h1_bridge PRIVATE -Wall -Wextra)
    # Optional operator UI (--ui / --manual).
    find_package(OpenCV QUIET COMPONENTS core imgproc highgui)
    if (OpenCV_FOUND)
        target_sources(h1_bridge PRIVATE ui.cpp)
        target_include_directories(h1_bridge PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(h1_bridge ${OpenCV_LIBS})
        target_compile_definitions(h1_bridge PRIVATE H1B_HAVE_UI)
        message(STATUS "h1_bridge: OpenCV ${OpenCV_VERSION} found, --ui / --manual enabled")
    else ()
        message(STATUS "h1_bridge: OpenCV not found, building without --ui / --manual")
    endif ()
    message(STATUS "h1_bridge: using unitree_sdk2 at ${UNITREE_SDK2_DIR}")
else ()
    message(WARNING "unitree_sdk2 not found at ${UNITREE_SDK2_DIR}: building the safety core + tests only")
endif ()
