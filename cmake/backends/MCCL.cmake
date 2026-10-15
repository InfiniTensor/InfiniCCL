include_guard(GLOBAL)

function(_infiniccl_find_mccl prefix provider required include_output library_output)
    if(provider STREQUAL "METAX")
        infiniccl_collect_metax_hints(_hints)
    elseif(provider STREQUAL "MOORE")
        infiniccl_collect_moore_hints(_hints)
    else()
        message(FATAL_ERROR "Unknown MCCL provider `${provider}`.")
    endif()

    set(_include_var "${prefix}_INC")
    set(_library_var "${prefix}_LIB")
    if(required)
        find_path(${_include_var} NAMES mccl.h
            HINTS ${_hints} PATH_SUFFIXES include REQUIRED)
        find_library(${_library_var} NAMES mccl
            HINTS ${_hints} PATH_SUFFIXES lib lib64 REQUIRED)
    else()
        find_path(${_include_var} NAMES mccl.h
            HINTS ${_hints} PATH_SUFFIXES include QUIET)
        find_library(${_library_var} NAMES mccl
            HINTS ${_hints} PATH_SUFFIXES lib lib64 QUIET)
    endif()

    set(${include_output} "${${_include_var}}" PARENT_SCOPE)
    set(${library_output} "${${_library_var}}" PARENT_SCOPE)
endfunction()

function(infiniccl_probe_mccl output)
    if(WITH_METAX)
        set(_provider METAX)
    elseif(WITH_MOORE)
        set(_provider MOORE)
    else()
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()

    _infiniccl_find_mccl(_INFINICCL_AUTO_MCCL ${_provider} FALSE _include _library)
    if(_include AND _library)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(infiniccl_configure_mccl include_output library_output)
    if(NOT WITH_METAX AND NOT WITH_MOORE)
        message(FATAL_ERROR
            "MCCL backend requires MetaX or Moore GPU support. "
            "Please enable `WITH_METAX` or `WITH_MOORE`.")
    endif()

    if(WITH_MOORE)
        set(_provider MOORE)
    else()
        set(_provider METAX)
    endif()
    _infiniccl_find_mccl(_INFINICCL_MCCL ${_provider} TRUE _include _library)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${library_output} "${_library}" PARENT_SCOPE)
endfunction()
