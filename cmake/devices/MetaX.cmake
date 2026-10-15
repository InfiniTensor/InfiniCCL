include_guard(GLOBAL)

function(infiniccl_collect_metax_hints output)
    set(_hints)
    if(DEFINED ENV{MACA_PATH} AND NOT "$ENV{MACA_PATH}" STREQUAL "")
        list(APPEND _hints "$ENV{MACA_PATH}")
    endif()
    list(APPEND _hints /opt/maca)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_metax output)
    set(_found FALSE)
    if(DEFINED ENV{MACA_PATH})
        set(_found TRUE)
    else()
        execute_process(
            COMMAND sh -c "grep -h 9999 /sys/bus/pci/devices/*/vendor 2>/dev/null"
            OUTPUT_VARIABLE _pci_vendor_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(FIND "${_pci_vendor_output}" "9999" _found_pos)
        if(_found_pos GREATER -1)
            set(_found TRUE)
        endif()
    endif()
    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_configure_metax)
    infiniccl_collect_metax_hints(_maca_hints)
    list(GET _maca_hints 0 _maca_path)
    if(NOT DEFINED ENV{MACA_PATH} OR "$ENV{MACA_PATH}" STREQUAL "")
        message(WARNING
            "MACA_PATH environment variable not set. Defaulting to /opt/maca/. "
            "Please verify this path is correct.")
    endif()

    get_filename_component(_wrapper
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/devices/mxcc_wrapper.sh" ABSOLUTE)
    include_directories("${_maca_path}/include")
    link_directories("${_maca_path}/lib")
    find_library(_INFINICCL_METAX_RUNTIME_LIB
        NAMES mcruntime HINTS "${_maca_path}/lib" REQUIRED)

    set(CMAKE_C_COMPILER "${_wrapper}" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER "${_wrapper}" PARENT_SCOPE)
    set(MACA_PATH "${_maca_path}" PARENT_SCOPE)
    set(MACA_RUNTIME_LIB "${_INFINICCL_METAX_RUNTIME_LIB}" PARENT_SCOPE)
endfunction()
