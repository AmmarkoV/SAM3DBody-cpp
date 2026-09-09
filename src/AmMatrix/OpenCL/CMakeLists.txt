project( matmul )
cmake_minimum_required( VERSION 2.8.7 )
 
#Make fast, lean and platform independent binaries..
set(CMAKE_CXX_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
set(CMAKE_C_FLAGS "-s -O3 -fPIC -march=native -mtune=native") 

OPTION(OPENCL_OPTIMIZATIONS OFF)

if (OPENCL_OPTIMIZATIONS)
add_definitions(-DOPENCL_OPTIMIZATIONS)
endif(OPENCL_OPTIMIZATIONS)

set_source_files_properties(mat_mul.cl PROPERTIES HEADER_FILE_ONLY TRUE)
#add_library(matmul STATIC mat_mul.c)
add_executable(matmul mat_mul.c )  
target_link_libraries(matmul m rt OpenCL )


set_target_properties(matmul PROPERTIES
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
