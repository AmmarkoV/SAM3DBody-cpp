# ---------------------------------------------------------------------------
# synchronization — QR timecode display tool (xqr_time_sync)
#
# Flashes fullscreen QR codes encoding live Unix timestamps so several cameras
# pointed at one screen can be aligned to a common clock in post (the multi-view
# pipeline reads these back — see MULTIVIEW_PLAN.md / README.md here).  Built
# from libQRSync.c (the libqrencode variant).
#
# Dependencies: libx11-dev, libxrandr-dev, libqrencode-dev (+ libm).  The target
# is optional — if any are missing it is skipped with a STATUS message so it
# never blocks the main build.
# ---------------------------------------------------------------------------

find_package(X11 QUIET)
find_path  (QRENCODE_INCLUDE_DIR qrencode.h)
find_library(QRENCODE_LIBRARY    qrencode)

# Optional tool: if anything is missing, name it + how to install, then bow out
# quietly (no error — it isn't used by the webcam.sh / video.sh front-ends).
set(_sync_missing "")
if(NOT X11_FOUND)
    list(APPEND _sync_missing libx11-dev)
endif()
if(NOT X11_Xrandr_FOUND)
    list(APPEND _sync_missing libxrandr-dev)
endif()
if(NOT QRENCODE_INCLUDE_DIR OR NOT QRENCODE_LIBRARY)
    list(APPEND _sync_missing libqrencode-dev)
endif()

if(_sync_missing)
    string(REPLACE ";" " " _sync_missing_str "${_sync_missing}")
    message(STATUS "synchronization: xqr_time_sync disabled — missing ${_sync_missing_str}")
    message(STATUS "                 optional QR time-sync tool; to enable: sudo apt install ${_sync_missing_str}")
    return()
endif()

add_executable(xqr_time_sync libQRSync.c)

target_include_directories(xqr_time_sync PRIVATE
    ${X11_INCLUDE_DIR}
    ${QRENCODE_INCLUDE_DIR}
)

target_link_libraries(xqr_time_sync PRIVATE
    ${X11_LIBRARIES}
    ${X11_Xrandr_LIB}
    ${QRENCODE_LIBRARY}
    m
)

set_target_properties(xqr_time_sync PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

target_compile_options(xqr_time_sync PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra -pedantic>
)

message(STATUS "  synchronization: xqr_time_sync (exe)")
