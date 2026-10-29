include_guard(GLOBAL)

function(infiniccl_collect_ascend_hints output)
    set(_hints)
    if(ASCEND_HOME)
        list(APPEND _hints "${ASCEND_HOME}")
    endif()
    foreach(_env ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_HOME)
        if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            list(APPEND _hints "$ENV{${_env}}")
        endif()
    endforeach()
    list(APPEND _hints /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_ascend output)
    set(_found FALSE)
    if(DEFINED ENV{ASCEND_HOME_PATH} OR DEFINED ENV{ASCEND_TOOLKIT_HOME}
       OR DEFINED ENV{ASCEND_HOME} OR EXISTS "/dev/davinci0"
       OR EXISTS "/dev/davinci_manager")
        set(_found TRUE)
    else()
        find_program(_INFINICCL_ASCEND_SMI_PATH npu-smi)
        if(_INFINICCL_ASCEND_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_ASCEND_SMI_PATH} info
                RESULT_VARIABLE _result
                OUTPUT_QUIET ERROR_QUIET TIMEOUT 15)
            if(_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
    endif()
    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_configure_ascend include_output library_output)
    infiniccl_collect_ascend_hints(_hints)

    find_path(_INFINICCL_ASCEND_INC NAMES acl/acl.h
        HINTS ${_hints} PATH_SUFFIXES include aarch64-linux/include x86_64-linux/include REQUIRED)
    find_library(_INFINICCL_ASCENDCL_LIB NAMES ascendcl
        HINTS ${_hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64 REQUIRED)
    # SDK FP16/BF16 types need `opdev` headers and `libnnopbase` for host conversions.
    find_path(_INFINICCL_ASCEND_OPDEV_INC NAMES opdev/bfloat16.h
        HINTS ${_hints} PATH_SUFFIXES include/aclnn aarch64-linux/include/aclnn x86_64-linux/include/aclnn REQUIRED)
    find_library(_INFINICCL_ASCEND_NNOPBASE_LIB NAMES nnopbase
        HINTS ${_hints} PATH_SUFFIXES lib64 aarch64-linux/lib64 x86_64-linux/lib64 REQUIRED)

    set(ASCEND_OPDEV_INC "${_INFINICCL_ASCEND_OPDEV_INC}" PARENT_SCOPE)
    set(ASCEND_NNOPBASE_LIB "${_INFINICCL_ASCEND_NNOPBASE_LIB}" PARENT_SCOPE)
    set(${include_output} "${_INFINICCL_ASCEND_INC}" PARENT_SCOPE)
    set(${library_output} "${_INFINICCL_ASCENDCL_LIB}" PARENT_SCOPE)
endfunction()
