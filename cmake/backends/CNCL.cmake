include_guard(GLOBAL)

function(_infiniccl_find_cncl prefix required include_output library_output)
    infiniccl_collect_cambricon_hints(_hints)
    set(_include_var "${prefix}_INC")
    set(_library_var "${prefix}_LIB")

    if(required)
        find_path(${_include_var} NAMES cncl.h
            HINTS ${_hints} PATH_SUFFIXES include REQUIRED)
        find_library(${_library_var} NAMES cncl
            HINTS ${_hints} PATH_SUFFIXES lib lib64 REQUIRED)
    else()
        find_path(${_include_var} NAMES cncl.h
            HINTS ${_hints} PATH_SUFFIXES include QUIET)
        find_library(${_library_var} NAMES cncl
            HINTS ${_hints} PATH_SUFFIXES lib lib64 QUIET)
    endif()

    set(${include_output} "${${_include_var}}" PARENT_SCOPE)
    set(${library_output} "${${_library_var}}" PARENT_SCOPE)
endfunction()

function(infiniccl_probe_cncl output)
    _infiniccl_find_cncl(_INFINICCL_AUTO_CNCL FALSE _include _library)
    if(_include AND _library)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(infiniccl_configure_cncl include_output library_output)
    if(NOT WITH_CAMBRICON)
        message(FATAL_ERROR "CNCL backend requires Cambricon device support. Please enable `WITH_CAMBRICON`.")
    endif()
    _infiniccl_find_cncl(_INFINICCL_CNCL TRUE _include _library)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${library_output} "${_library}" PARENT_SCOPE)
endfunction()
