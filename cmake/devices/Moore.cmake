include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/../architecture/MusaArchitecture.cmake")

function(infiniccl_collect_moore_hints output)
    set(_hints)
    foreach(_env MUSA_ROOT MUSA_HOME MUSA_PATH)
        if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            list(APPEND _hints "$ENV{${_env}}")
        endif()
    endforeach()
    list(APPEND _hints /usr/local/musa)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_moore output)
    set(_found FALSE)
    file(GLOB _device_files "/dev/mtgpu*")

    if(DEFINED ENV{MUSA_ROOT} OR DEFINED ENV{MUSA_HOME} OR DEFINED ENV{MUSA_PATH})
        set(_found TRUE)
    elseif(_device_files)
        set(_found TRUE)
    else()
        find_program(_INFINICCL_MOORE_SMI_PATH mthreads-gmi)
        if(_INFINICCL_MOORE_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_MOORE_SMI_PATH} -L
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

function(infiniccl_configure_moore)
    set(_musa_root)
    foreach(_env MUSA_ROOT MUSA_HOME MUSA_PATH)
        if(NOT _musa_root AND DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            set(_musa_root "$ENV{${_env}}")
        endif()
    endforeach()

    if(NOT _musa_root AND EXISTS "/usr/local/musa")
        set(_musa_root "/usr/local/musa")
    endif()

    if(NOT _musa_root)
        message(FATAL_ERROR
            "`WITH_MOORE` is `ON` but `MUSA_ROOT`/`MUSA_HOME`/`MUSA_PATH` "
            "is not set and `/usr/local/musa` was not found.")
    endif()
    if(NOT EXISTS "${_musa_root}/bin/mcc")
        message(FATAL_ERROR "Could not find `mcc` under `${_musa_root}/bin`.")
    endif()

    get_filename_component(_wrapper
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/devices/mcc_wrapper.sh" ABSOLUTE)
    file(CHMOD "${_wrapper}"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

    include_directories("${_musa_root}/include")
    link_directories("${_musa_root}/lib")
    find_library(_INFINICCL_MUSA_LIB NAMES musa HINTS "${_musa_root}/lib" REQUIRED)
    find_library(_INFINICCL_MUSART_LIB NAMES musart HINTS "${_musa_root}/lib" REQUIRED)
    find_library(_INFINICCL_MUBLAS_LIB NAMES mublas HINTS "${_musa_root}/lib" REQUIRED)

    infiniccl_resolve_musa_architecture_config(
        "${_musa_root}/include"
        "${_INFINICCL_MUSART_LIB}"
    )

    set(CMAKE_C_COMPILER "${_wrapper}" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER "${_wrapper}" PARENT_SCOPE)
    set(MUSA_ROOT "${_musa_root}" PARENT_SCOPE)
    set(MUSA_LIB "${_INFINICCL_MUSA_LIB}" PARENT_SCOPE)
    set(MUSART_LIB "${_INFINICCL_MUSART_LIB}" PARENT_SCOPE)
    set(MUBLAS_LIB "${_INFINICCL_MUBLAS_LIB}" PARENT_SCOPE)
    set(MUSA_ARCHITECTURES "${MUSA_ARCHITECTURES}" PARENT_SCOPE)
    set(MUSA_MARCH_TYPE "${MUSA_MARCH_TYPE}" PARENT_SCOPE)
    set(MUSA_ARCH_COMPILE_OPTIONS "${MUSA_ARCH_COMPILE_OPTIONS}" PARENT_SCOPE)
endfunction()
