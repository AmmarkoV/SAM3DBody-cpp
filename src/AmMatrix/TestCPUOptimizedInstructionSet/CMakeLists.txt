project( TestCPUOptimizedInstructionSet ) 
cmake_minimum_required( VERSION 2.8.7 )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

 
add_executable(TestCPUOptimizedInstructionSet main.c ../matrix4x4Tools.c )  
target_link_libraries(TestCPUOptimizedInstructionSet rt dl m AmMatrix )
set_target_properties(TestCPUOptimizedInstructionSet PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(TestCPUOptimizedInstructionSet AmMatrix) 

set_target_properties(TestCPUOptimizedInstructionSet PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
