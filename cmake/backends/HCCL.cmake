include_guard(GLOBAL)

function(_infiniccl_find_hccl prefix required include_output library_output)
    infiniccl_collect_ascend_hints(_hints)
    set(_include_var "${prefix}_INC")
    set(_library_var "${prefix}_LIB")

    if(required)
        find_path(${_include_var} NAMES hccl/hccl.h
            HINTS ${_hints} PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include REQUIRED)
        find_library(${_library_var} NAMES hccl
            HINTS ${_hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64 REQUIRED)
    else()
        find_path(${_include_var} NAMES hccl/hccl.h
            HINTS ${_hints} PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include QUIET)
        find_library(${_library_var} NAMES hccl
            HINTS ${_hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64 QUIET)
    endif()

    set(${include_output} "${${_include_var}}" PARENT_SCOPE)
    set(${library_output} "${${_library_var}}" PARENT_SCOPE)
endfunction()

function(infiniccl_probe_hccl output)
    _infiniccl_find_hccl(_INFINICCL_AUTO_HCCL FALSE _include _library)
    if(_include AND _library)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(infiniccl_configure_hccl include_output library_output)
    if(NOT WITH_ASCEND)
        message(FATAL_ERROR "HCCL backend requires Ascend device support. Please enable `WITH_ASCEND`.")
    endif()
    _infiniccl_find_hccl(_INFINICCL_HCCL TRUE _include _library)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${library_output} "${_library}" PARENT_SCOPE)
endfunction()
