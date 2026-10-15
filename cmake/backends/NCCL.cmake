include_guard(GLOBAL)

function(_infiniccl_find_nccl prefix required include_output library_output definitions_output)
    set(_hints)
    set(_header_names nccl.h)
    set(_library_names nccl)
    set(_include_suffixes include)
    set(_library_suffixes lib lib64)
    set(_definitions)

    if(WITH_THEAD)
        if(_THEAD_CUDA_HINTS)
            set(_hints ${_THEAD_CUDA_HINTS})
        else()
            infiniccl_collect_thead_hints(_hints)
        endif()
        list(APPEND _include_suffixes targets/x86_64-linux/include)
        list(APPEND _library_suffixes targets/x86_64-linux/lib)
        message(STATUS "T-Head detected. Searching for NCCL in ${_hints}")
    elseif(WITH_ILUVATAR)
        set(_hints /usr/local/corex)
        message(STATUS "Iluvatar detected. Searching for NCCL in ${_hints}")
    elseif(WITH_HYGON)
        infiniccl_collect_hygon_hints(_hints)
        if(HYGON_DTK_ROOT)
            list(PREPEND _hints "${HYGON_DTK_ROOT}")
        endif()
        set(_roots ${_hints})
        foreach(_root IN LISTS _roots)
            list(APPEND _hints "${_root}/rccl")
        endforeach()
        set(_header_names rccl/rccl.h rccl.h)
        set(_library_names rccl)
        set(_definitions INFINI_CCL_USE_RCCL)
        list(APPEND _library_suffixes rccl/lib)
        message(STATUS "Hygon detected. Searching for NCCL-compatible RCCL in ${_hints}")
    endif()
    list(REMOVE_DUPLICATES _hints)

    set(_include_var "${prefix}_INC")
    set(_library_var "${prefix}_LIB")
    if(required)
        find_path(${_include_var} NAMES ${_header_names}
            HINTS ${_hints}
            PATH_SUFFIXES ${_include_suffixes}
            REQUIRED)
        find_library(${_library_var} NAMES ${_library_names}
            HINTS ${_hints}
            PATH_SUFFIXES ${_library_suffixes}
            REQUIRED)
    else()
        find_path(${_include_var} NAMES ${_header_names}
            HINTS ${_hints}
            PATH_SUFFIXES ${_include_suffixes}
            QUIET)
        find_library(${_library_var} NAMES ${_library_names}
            HINTS ${_hints}
            PATH_SUFFIXES ${_library_suffixes}
            QUIET)
    endif()

    set(${include_output} "${${_include_var}}" PARENT_SCOPE)
    set(${library_output} "${${_library_var}}" PARENT_SCOPE)
    set(${definitions_output} "${_definitions}" PARENT_SCOPE)
endfunction()

function(infiniccl_probe_nccl output)
    _infiniccl_find_nccl(_INFINICCL_AUTO_NCCL FALSE _include _library _definitions)
    if(_include AND _library)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(infiniccl_configure_nccl include_output library_output definitions_output)
    if(NOT WITH_NVIDIA AND NOT WITH_THEAD AND NOT WITH_ILUVATAR AND NOT WITH_HYGON)
        message(FATAL_ERROR
            "NCCL backend requires NVIDIA, T-Head, Iluvatar, or Hygon GPU support. "
            "Please enable `WITH_NVIDIA`, `WITH_THEAD`, `WITH_ILUVATAR`, or `WITH_HYGON`.")
    endif()

    _infiniccl_find_nccl(_INFINICCL_NCCL TRUE _include _library _definitions)
    set(${include_output} "${_include}" PARENT_SCOPE)
    set(${library_output} "${_library}" PARENT_SCOPE)
    set(${definitions_output} "${_definitions}" PARENT_SCOPE)
endfunction()
