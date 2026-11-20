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

function(infiniccl_configure_mpi_device_aware output)
    if(NOT WITH_OMPI)
        set(${output} "" PARENT_SCOPE)
        return()
    endif()

    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)

    set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
    if(MPI_CXX_COMPILE_FLAGS)
        set(CMAKE_REQUIRED_FLAGS ${MPI_CXX_COMPILE_FLAGS})
    endif()
    if(MPI_CXX_LINK_FLAGS)
        set(CMAKE_REQUIRED_LINK_OPTIONS ${MPI_CXX_LINK_FLAGS})
    endif()

    check_cxx_source_compiles("
        #include <mpi.h>
        #include <mpi-ext.h>
        int main() {
            return MPIX_Query_cuda_support();
        }
    " INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
    cmake_pop_check_state()

    if(INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
        set(${output} "INFINICCL_MPI_HAS_OMPI_CUDA_QUERY=1" PARENT_SCOPE)
    else()
        set(${output} "" PARENT_SCOPE)
    endif()
endfunction()
