include_guard(GLOBAL)

function(_infiniccl_find_hccl_average prefix required hints include_output library_output)
    set(_include_var "${prefix}_ACLNN_INC")
    set(_library_var "${prefix}_OPAPI_LIB")
    set(_find_mode "")
    if(required)
        set(_find_mode REQUIRED)
    endif()

    find_path(${_include_var} NAMES aclnnop/aclnn_div.h
        HINTS ${hints} PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include ${_find_mode})
    find_library(${_library_var} NAMES opapi
        HINTS ${hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64 ${_find_mode})

    set(${include_output} "${${_include_var}}" PARENT_SCOPE)
    set(${library_output} "${${_library_var}}" PARENT_SCOPE)
endfunction()

function(_infiniccl_find_hccl prefix required include_output library_output aclnn_include_output opapi_library_output)
    infiniccl_collect_ascend_hints(_hints)
    set(_include_var "${prefix}_INC")
    set(_library_var "${prefix}_LIB")
    set(_aclnn_include "")
    set(_opapi_library "")

    if(required)
        find_path(${_include_var} NAMES hccl/hccl.h
            HINTS ${_hints} PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include REQUIRED)
        find_library(${_library_var} NAMES hccl
            HINTS ${_hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64 REQUIRED)
        # AVG dependencies are required for configuration, not basic HCCL probing.
        _infiniccl_find_hccl_average(${prefix} TRUE "${_hints}"
            _aclnn_include _opapi_library)
    else()
        find_path(${_include_var} NAMES hccl/hccl.h
            HINTS ${_hints} PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include)
        find_library(${_library_var} NAMES hccl
            HINTS ${_hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64)
    endif()

    set(${include_output} "${${_include_var}}" PARENT_SCOPE)
    set(${library_output} "${${_library_var}}" PARENT_SCOPE)
    set(${aclnn_include_output} "${_aclnn_include}" PARENT_SCOPE)
    set(${opapi_library_output} "${_opapi_library}" PARENT_SCOPE)
endfunction()

function(infiniccl_probe_hccl output)
    _infiniccl_find_hccl(_INFINICCL_AUTO_HCCL FALSE
        _include _library _aclnn_include _opapi_library)
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

    _infiniccl_find_hccl(_INFINICCL_HCCL TRUE
        _include _library _aclnn_include _opapi_library)

    set(HCCL_ACLNN_INC "${_aclnn_include}" PARENT_SCOPE)
    set(HCCL_OPAPI_LIB "${_opapi_library}" PARENT_SCOPE)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${library_output} "${_library}" PARENT_SCOPE)
endfunction()
