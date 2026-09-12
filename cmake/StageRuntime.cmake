# A static dependency configuration can legitimately produce an empty list.
foreach(_dll IN LISTS SAM3D_RUNTIME_DLLS)
    if(NOT _dll STREQUAL "")
        get_filename_component(_name "${_dll}" NAME)
        file(COPY_FILE "${_dll}" "${SAM3D_DESTINATION}/${_name}" ONLY_IF_DIFFERENT)
    endif()
endforeach()
