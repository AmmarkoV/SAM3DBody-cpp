# ---------------------------------------------------------------------------
# multiview — multi-camera capture & fusion (see MULTIVIEW_PLAN.md)
#
# So far: calib — a self-contained Stereolabs/Matlab `.calib` reader/writer
# (intrinsics + distortion + extrinsics).  No external dependencies.
# ---------------------------------------------------------------------------

add_library(multiview_calib STATIC calib.c)

target_include_directories(multiview_calib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(multiview_calib PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

# Unit test: parse the committed fixture and check the values.
add_executable(calib_test calib_test.c)
target_link_libraries(calib_test PRIVATE multiview_calib m)
add_test(NAME calib_test
         COMMAND calib_test ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample.calib)

# ---------------------------------------------------------------------------
# aruco_qr — ArUco pose + QR timecode detection (needs the aruco/objdetect
# OpenCV modules; requested here so the core targets keep their own component
# set unchanged).
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d aruco objdetect videoio)

add_library(multiview_aruco_qr STATIC aruco_qr.cpp)
target_include_directories(multiview_aruco_qr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(multiview_aruco_qr PUBLIC multiview_calib ${OpenCV_LIBS})
target_compile_options(multiview_aruco_qr PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

add_executable(aruco_qr_probe aruco_qr_probe.cpp)
target_link_libraries(aruco_qr_probe PRIVATE multiview_aruco_qr ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# extrinsics — camera placement from shared static markers (pure C++ math) +
# a synthetic unit test + the driver that runs detection over the videos.
# ---------------------------------------------------------------------------
add_library(multiview_extrinsics STATIC extrinsics.cpp)
target_include_directories(multiview_extrinsics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(multiview_extrinsics PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

add_executable(extrinsics_test extrinsics_test.cpp)
target_link_libraries(extrinsics_test PRIVATE multiview_extrinsics)
add_test(NAME extrinsics_test COMMAND extrinsics_test)

add_executable(sam_3dbody_extrinsics sam_3dbody_extrinsics.cpp)
target_link_libraries(sam_3dbody_extrinsics PRIVATE
    multiview_extrinsics multiview_aruco_qr multiview_calib ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# sync_time — per-stream QR-timecode -> common clock fit (pure C++) + test +
# the driver that extracts QR timestamps from a video and fits them.
# ---------------------------------------------------------------------------
add_library(multiview_sync_time STATIC sync_time.cpp)
target_include_directories(multiview_sync_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(multiview_sync_time PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

add_executable(sync_time_test sync_time_test.cpp)
target_link_libraries(sync_time_test PRIVATE multiview_sync_time)
add_test(NAME sync_time_test COMMAND sync_time_test)

add_executable(sam_3dbody_synctime sam_3dbody_synctime.cpp)
target_link_libraries(sam_3dbody_synctime PRIVATE
    multiview_sync_time multiview_aruco_qr ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# sync_io — read/write the session .sync file (pure C++) + round-trip test +
# sam_3dbody_sync, which fuses extrinsics + temporal fit into one session.sync.
# ---------------------------------------------------------------------------
add_library(multiview_sync_io STATIC sync_io.cpp)
target_include_directories(multiview_sync_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multiview_sync_io PUBLIC multiview_extrinsics)   # mv::Mat4
target_compile_options(multiview_sync_io PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

add_executable(sync_io_test sync_io_test.cpp)
target_link_libraries(sync_io_test PRIVATE multiview_sync_io)
add_test(NAME sync_io_test COMMAND sync_io_test)

add_executable(sam_3dbody_sync sam_3dbody_sync.cpp)
target_link_libraries(sam_3dbody_sync PRIVATE
    multiview_sync_io multiview_sync_time multiview_extrinsics
    multiview_aruco_qr multiview_calib ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# fusion front-end — per-stream offline inference driven by a .sync (reuses the
# offline passes from the fast_sam_3dbody lib).  Fuser/association on top later.
# ---------------------------------------------------------------------------
add_library(multiview_frontend STATIC multiview_frontend.cpp)
target_include_directories(multiview_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multiview_frontend PUBLIC
    multiview_sync_io fast_sam_3dbody ${OpenCV_LIBS})

add_executable(sam_3dbody_multiview sam_3dbody_multiview.cpp)
target_link_libraries(sam_3dbody_multiview PRIVATE
    multiview_frontend multiview_sync_io fast_sam_3dbody ${OpenCV_LIBS})

message(STATUS "  multiview: calib + aruco_qr + extrinsics + sync_time + sync_io (libs); tests: calib/extrinsics/sync_time/sync_io; tools: aruco_qr_probe, sam_3dbody_extrinsics, sam_3dbody_synctime, sam_3dbody_sync")
