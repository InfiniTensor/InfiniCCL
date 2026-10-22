include_guard(GLOBAL)

function(infiniccl_collect_thead_hints output)
    get_property(_hints GLOBAL PROPERTY INFINICCL_THEAD_HINTS)
    if(NOT _hints)
        set(_hints
            /usr/local/PPU_SDK
            /usr/local/PPU_SDK/CUDA_SDK
        )

        foreach(_env IN ITEMS PPU_SDK PPU_HOME PPU_PATH)
            if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
                list(APPEND _hints
                    "$ENV{${_env}}"
                    "$ENV{${_env}}/CUDA_SDK"
                )
            endif()
        endforeach()

        foreach(_env IN ITEMS CUDA_SDK CUDA_HOME CUDA_PATH CUDA_TOOLKIT_ROOT)
            if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
                list(APPEND _hints
                    "$ENV{${_env}}"
                    "$ENV{${_env}}/.."
                )
            endif()
        endforeach()

        list(REMOVE_DUPLICATES _hints)
        set_property(GLOBAL PROPERTY INFINICCL_THEAD_HINTS "${_hints}")
    endif()
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_thead output)
    set(_found FALSE)
    infiniccl_collect_thead_hints(_hints)
    foreach(_hint IN LISTS _hints)
        if(EXISTS "${_hint}/CUDA_SDK"
           OR EXISTS "${_hint}/bin/ppu-smi"
           OR EXISTS "${_hint}/ppu-smi/bin/ppu-smi")
            set(_found TRUE)
        endif()
    endforeach()

    if(NOT _found)
        find_program(_INFINICCL_THEAD_SMI_PATH
            NAMES ppu-smi nvidia-smi
            HINTS /usr/local/PPU_SDK/bin /usr/local/PPU_SDK/ppu-smi/bin
        )
        if(_INFINICCL_THEAD_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_THEAD_SMI_PATH} -L
                RESULT_VARIABLE _smi_result
                OUTPUT_VARIABLE _smi_output
                ERROR_QUIET
            )
            if(_smi_result EQUAL 0 AND _smi_output MATCHES "PPU")
                set(_found TRUE)
            endif()
        endif()
    endif()
    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_resolve_thead_cuda root_output hints_output)
    infiniccl_collect_thead_hints(_hints)
    list(APPEND _hints /usr/local/cuda)
    list(REMOVE_DUPLICATES _hints)

    set(_cuda_root)
    foreach(_hint IN LISTS _hints)
        if(NOT _cuda_root AND EXISTS "${_hint}/include/cuda_runtime.h")
            set(_cuda_root "${_hint}")
        endif()
    endforeach()
    if(NOT _cuda_root)
        message(FATAL_ERROR
            "`WITH_THEAD` is `ON` but the T-Head CUDA-compatible SDK was not found. "
            "Set `PPU_SDK`, `PPU_HOME`, `CUDA_SDK`, `CUDA_HOME`, or `CUDA_PATH`.")
    endif()

    set(${root_output} "${_cuda_root}" PARENT_SCOPE)
    set(${hints_output} "${_hints}" PARENT_SCOPE)
endfunction()
