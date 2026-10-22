include_guard(GLOBAL)

function(infiniccl_collect_cambricon_hints output)
    set(_hints)
    if(DEFINED ENV{NEUWARE_HOME} AND NOT "$ENV{NEUWARE_HOME}" STREQUAL "")
        list(APPEND _hints "$ENV{NEUWARE_HOME}")
    endif()
    list(APPEND _hints /usr/local/neuware)
    list(REMOVE_DUPLICATES _hints)
    set(${output} "${_hints}" PARENT_SCOPE)
endfunction()

function(infiniccl_detect_cambricon output)
    set(_found FALSE)
    if(DEFINED ENV{NEUWARE_HOME})
        set(_found TRUE)
    else()
        find_program(_INFINICCL_CAMBRICON_CNMON_PATH cnmon)
        if(_INFINICCL_CAMBRICON_CNMON_PATH)
            execute_process(
                COMMAND ${_INFINICCL_CAMBRICON_CNMON_PATH} -l
                RESULT_VARIABLE _cnmon_result
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_cnmon_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
    endif()
    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_configure_cambricon)
    set(_neuware_home "$ENV{NEUWARE_HOME}")

    include_directories("${_neuware_home}/include")
    link_directories("${_neuware_home}/lib" "${_neuware_home}/lib64")
    find_library(_INFINICCL_CAMBRICON_RUNTIME_LIB
        NAMES cnrt HINTS "${_neuware_home}/lib64" REQUIRED)
    find_library(_INFINICCL_CAMBRICON_CNNL_LIB
        NAMES cnnl HINTS "${_neuware_home}/lib64" REQUIRED)
    find_library(_INFINICCL_CAMBRICON_CNNL_EXTRA_LIB
        NAMES cnnl_extra HINTS "${_neuware_home}/lib64" REQUIRED)
    find_library(_INFINICCL_CAMBRICON_PAPI_LIB
        NAMES cnpapi HINTS "${_neuware_home}/lib64" REQUIRED)

    set(NEUWARE_HOME "${_neuware_home}" PARENT_SCOPE)
    set(CAMBRICON_RUNTIME_LIB "${_INFINICCL_CAMBRICON_RUNTIME_LIB}" PARENT_SCOPE)
    set(CAMBRICON_CNNL_LIB "${_INFINICCL_CAMBRICON_CNNL_LIB}" PARENT_SCOPE)
    set(CAMBRICON_CNNL_EXTRA_LIB "${_INFINICCL_CAMBRICON_CNNL_EXTRA_LIB}" PARENT_SCOPE)
    set(CAMBRICON_PAPI_LIB "${_INFINICCL_CAMBRICON_PAPI_LIB}" PARENT_SCOPE)
endfunction()
