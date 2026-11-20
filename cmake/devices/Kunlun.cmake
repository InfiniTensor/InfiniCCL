include_guard(GLOBAL)

function(infiniccl_collect_kunlun_hints output)
    set(_hints)
    if(KUNLUN_HOME)
        list(APPEND _hints "${KUNLUN_HOME}")
    endif()
    if(DEFINED ENV{KUNLUN_HOME} AND NOT "$ENV{KUNLUN_HOME}" STREQUAL "")
        list(APPEND _hints "$ENV{KUNLUN_HOME}")
    endif()
    list(APPEND _hints /usr/local/xtdk)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_kunlun output)
    set(_found FALSE)
    file(GLOB _KUNLUN_DEV_FILES "/dev/xpu*")
    if(_KUNLUN_DEV_FILES)
        set(_found TRUE)
    else()
        find_program(_INFINICCL_KUNLUN_SMI_PATH xpu-smi)
        if(_INFINICCL_KUNLUN_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_KUNLUN_SMI_PATH}
                RESULT_VARIABLE _result
                OUTPUT_QUIET ERROR_QUIET TIMEOUT 15)
            if(_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
    endif()
    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_configure_kunlun include_output runtime_library_output)
    infiniccl_collect_kunlun_hints(_hints)

    # `xccl.h` (communication API) ships in the XTDK `xccl/` tree; the XRE
    # runtime headers (`xpu/runtime.h`) in `xre/`.
    find_path(_INFINICCL_BKCL_INC NAMES xccl.h
        HINTS ${_hints} PATH_SUFFIXES xccl/include include REQUIRED)
    find_path(_INFINICCL_XRE_INC NAMES xpu/runtime.h
        HINTS ${_hints} PATH_SUFFIXES xre/include include REQUIRED)
    find_library(_INFINICCL_XPURT_LIB NAMES xpurt
        HINTS ${_hints} PATH_SUFFIXES xre/so lib lib64 REQUIRED)

    set(${include_output} "${_INFINICCL_BKCL_INC};${_INFINICCL_XRE_INC}"
        PARENT_SCOPE)
    set(${runtime_library_output} "${_INFINICCL_XPURT_LIB}" PARENT_SCOPE)
endfunction()
