include_guard(GLOBAL)

function(infiniccl_probe_bkcl output)
    infiniccl_collect_kunlun_hints(_hints)
    find_path(_INFINICCL_AUTO_BKCL_INC NAMES xccl.h
        HINTS ${_hints} PATH_SUFFIXES xccl/include include)
    find_library(_INFINICCL_AUTO_BKCL_LIB NAMES bkcl
        HINTS ${_hints} PATH_SUFFIXES lib lib64)
    if(_INFINICCL_AUTO_BKCL_INC AND _INFINICCL_AUTO_BKCL_LIB)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(infiniccl_configure_bkcl include_output library_output)
    if(NOT WITH_KUNLUN)
        message(FATAL_ERROR "BKCL backend requires Kunlun XPU support. Please enable `WITH_KUNLUN`.")
    endif()

    infiniccl_collect_kunlun_hints(_hints)
    find_path(_INFINICCL_BKCL_INC NAMES xccl.h
        HINTS ${_hints} PATH_SUFFIXES xccl/include include REQUIRED)
    find_library(_INFINICCL_BKCL_LIB NAMES bkcl
        HINTS ${_hints} PATH_SUFFIXES lib lib64 REQUIRED)

    set(${include_output} "${_INFINICCL_BKCL_INC}" PARENT_SCOPE)
    set(${library_output} "${_INFINICCL_BKCL_LIB}" PARENT_SCOPE)
endfunction()
