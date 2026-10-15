include_guard(GLOBAL)

function(infiniccl_detect_nvidia output)
    set(_found FALSE)
    file(GLOB _device_files "/dev/nvidia0")

    if(_device_files)
        set(_found TRUE)
    else()
        find_program(_INFINICCL_NVIDIA_SMI_PATH nvidia-smi)
        if(_INFINICCL_NVIDIA_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_NVIDIA_SMI_PATH} -L
                RESULT_VARIABLE _smi_result
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_smi_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
    endif()

    set(${output} ${_found} PARENT_SCOPE)
endfunction()
