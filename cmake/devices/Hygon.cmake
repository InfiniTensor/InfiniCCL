include_guard(GLOBAL)

function(infiniccl_collect_hygon_hints output)
    set(_hints)
    foreach(_env DTKROOT DTK_ROOT ROCM_PATH)
        if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            list(APPEND _hints "$ENV{${_env}}")
        endif()
    endforeach()
    if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        get_filename_component(_root_from_hip "$ENV{HIP_PATH}/.." ABSOLUTE)
        list(APPEND _hints "${_root_from_hip}")
    endif()
    list(APPEND _hints /opt/dtk)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_hygon output)
    set(_found FALSE)
    if((DEFINED ENV{DTKROOT} OR DEFINED ENV{DTK_ROOT} OR DEFINED ENV{ROCM_PATH})
       AND EXISTS "/dev/kfd")
        set(_found TRUE)
    elseif(EXISTS "/opt/dtk" AND EXISTS "/dev/kfd")
        set(_found TRUE)
    else()
        find_program(_INFINICCL_HYGON_SMI_PATH
            NAMES hy-smi rocm-smi
            HINTS /opt/hyhal/bin /opt/dtk/bin
        )
        if(_INFINICCL_HYGON_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_HYGON_SMI_PATH}
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

function(infiniccl_configure_hygon)
    set(_dtk_root)
    foreach(_env DTKROOT DTK_ROOT ROCM_PATH)
        if(NOT _dtk_root AND DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            set(_dtk_root "$ENV{${_env}}")
        endif()
    endforeach()

    if(NOT _dtk_root AND DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        get_filename_component(_dtk_root "$ENV{HIP_PATH}/.." ABSOLUTE)
    endif()

    if(NOT _dtk_root AND EXISTS "/opt/dtk")
        set(_dtk_root "/opt/dtk")
    endif()
    if(NOT _dtk_root)
        message(FATAL_ERROR
            "`WITH_HYGON` is `ON` but DTK was not found. "
            "Set `DTKROOT`, `DTK_ROOT`, `ROCM_PATH`, or `HIP_PATH`.")
    endif()

    set(_hip_hints "${_dtk_root}" "${_dtk_root}/hip")
    find_path(_INFINICCL_HYGON_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime.h
        HINTS ${_hip_hints}
        PATH_SUFFIXES include hip/include
        REQUIRED
    )
    find_library(_INFINICCL_HYGON_HIP_RUNTIME_LIB
        NAMES amdhip64
        HINTS ${_hip_hints}
        PATH_SUFFIXES lib lib64 hip/lib
        REQUIRED
    )
    find_program(_INFINICCL_HYGON_HIPCC_COMPILER
        NAMES hipcc
        HINTS ${_hip_hints}
        PATH_SUFFIXES bin hip/bin
        REQUIRED
    )

    set(CMAKE_CXX_COMPILER "${_INFINICCL_HYGON_HIPCC_COMPILER}"
        CACHE FILEPATH "Hygon DTK hipcc compiler" FORCE)
    include_directories(${_INFINICCL_HYGON_HIP_INCLUDE_DIR})

    set(HYGON_DTK_ROOT "${_dtk_root}" PARENT_SCOPE)
    set(HYGON_HIP_INCLUDE_DIR "${_INFINICCL_HYGON_HIP_INCLUDE_DIR}" PARENT_SCOPE)
    set(HYGON_HIP_RUNTIME_LIB "${_INFINICCL_HYGON_HIP_RUNTIME_LIB}" PARENT_SCOPE)
    set(HYGON_HIPCC_COMPILER "${_INFINICCL_HYGON_HIPCC_COMPILER}" PARENT_SCOPE)
    set(HYGON_HIP_DEFINITIONS __HIP_PLATFORM_AMD__ PARENT_SCOPE)
endfunction()
