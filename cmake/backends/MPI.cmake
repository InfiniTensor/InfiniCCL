include_guard(GLOBAL)

function(infiniccl_detect_mpi output)
    set(_implementation)
    find_program(_INFINICCL_MPIRUN_PATH NAMES mpirun mpiexec)
    find_program(_INFINICCL_MPICC_PATH NAMES mpicc)

    if(_INFINICCL_MPIRUN_PATH AND _INFINICCL_MPICC_PATH)
        execute_process(
            COMMAND ${_INFINICCL_MPIRUN_PATH} --version
            OUTPUT_VARIABLE _version_out
            ERROR_VARIABLE _version_err
        )
        set(_version_info "${_version_out}${_version_err}")

        if(_version_info MATCHES "Open MPI" OR _version_info MATCHES "PRRTE")
            set(_implementation OMPI)
        elseif(_version_info MATCHES "MPICH"
               OR _version_info MATCHES "Hydra"
               OR _version_info MATCHES "Intel\\(R\\) MPI")
            set(_implementation MPICH)
        else()
            set(_implementation OMPI)
            message(STATUS "Unknown MPI implementation. Defaulting to `WITH_OMPI=ON`")
        endif()
    endif()

    set(${output} "${_implementation}" PARENT_SCOPE)
endfunction()
