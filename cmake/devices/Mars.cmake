include_guard(GLOBAL)

function(infiniccl_collect_mars_hints output)
    set(_hints)
    if(DEFINED ENV{HPCC_PATH} AND NOT "$ENV{HPCC_PATH}" STREQUAL "")
        list(APPEND _hints "$ENV{HPCC_PATH}")
    endif()
    list(APPEND _hints /opt/hpcc)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_mars output)
    set(_found FALSE)
    if(DEFINED ENV{HPCC_PATH} AND NOT "$ENV{HPCC_PATH}" STREQUAL "")
        set(_found TRUE)
    endif()
    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_configure_mars)
    infiniccl_collect_mars_hints(_hints)
    find_path(_INFINICCL_HPCC_INC NAMES hcr/hc_runtime.h
        HINTS ${_hints} PATH_SUFFIXES include REQUIRED)
    find_library(_INFINICCL_HPCC_RUNTIME_LIB NAMES hcruntime
        HINTS ${_hints} PATH_SUFFIXES lib lib64 REQUIRED)
    include_directories("${_INFINICCL_HPCC_INC}")
    set(HPCC_RUNTIME_LIB "${_INFINICCL_HPCC_RUNTIME_LIB}" PARENT_SCOPE)
endfunction()
