include_guard(GLOBAL)

# Preserve explicitly enabled device options and probe only devices that have
# not been requested. The caller retains responsibility for cross-device
# policies after all individual device results are available.
function(infiniccl_resolve_device_detection option display_name detector result_output)
    if(NOT COMMAND ${detector})
        message(FATAL_ERROR "Unknown device detector `${detector}` for `${display_name}`.")
    endif()

    set(_requested ${${option}})
    if(_requested)
        set(_found TRUE)
        message(STATUS "${display_name} support explicitly enabled.")
    else()
        cmake_language(CALL ${detector} _found)
        if(_found)
            message(STATUS "Auto-detected ${display_name} environment.")
        else()
            message(STATUS "${display_name} environment not detected.")
        endif()
    endif()

    if(_found)
        set(${option} ON PARENT_SCOPE)
    endif()
    set(${result_output} ${_found} PARENT_SCOPE)
endfunction()
