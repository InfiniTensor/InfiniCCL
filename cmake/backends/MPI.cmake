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
    set(_supported 0)
    set(_probe_includes "")
    set(_probe_libraries "")
    set(_probe_compile_flags "")
    set(_probe_link_flags "")
    set(_probe_library_files "")

    if(WITH_OMPI)
        set(_probe_includes "${MPI_CXX_INCLUDE_DIRS}")
        if(TARGET MPI::MPI_CXX)
            set(_probe_libraries MPI::MPI_CXX)
            get_target_property(_mpi_imported_locations MPI::MPI_CXX
                IMPORTED_LOCATION)
            if(_mpi_imported_locations)
                list(APPEND _probe_library_files ${_mpi_imported_locations})
            endif()
        else()
            set(_probe_libraries "${MPI_CXX_LIBRARIES}")
            set(_probe_library_files "${_probe_libraries}")
            set(_probe_compile_flags "${MPI_CXX_COMPILE_FLAGS}")
            set(_probe_link_flags "${MPI_CXX_LINK_FLAGS}")
        endif()
    endif()

    set(_fingerprint
        "gen=1|implementation=${WITH_OMPI}|includes=${_probe_includes}|libraries=${_probe_libraries}|compile_flags=${_probe_compile_flags}|link_flags=${_probe_link_flags}")
    foreach(_library IN LISTS _probe_library_files)
        if(EXISTS "${_library}")
            set(_library_mtime "")
            file(TIMESTAMP "${_library}" _library_mtime "%s")
            string(APPEND _fingerprint "|mtime:${_library}=${_library_mtime}")
        endif()
    endforeach()
    string(MD5 _fingerprint_md5 "${_fingerprint}")

    set(_reuse FALSE)
    set(_cached_supported "")
    set(_stamp
        "${CMAKE_BINARY_DIR}/src/backends/mpi/ompi/device_support_config.stamp")
    if(EXISTS "${_stamp}")
        file(READ "${_stamp}" _stamp_text)
        string(REPLACE "\n" ";" _stamp_lines "${_stamp_text}")
        list(GET _stamp_lines 0 _stamp_md5)
        list(LENGTH _stamp_lines _stamp_line_count)
        if(_stamp_line_count GREATER 1
                AND "${_stamp_md5}" STREQUAL "${_fingerprint_md5}")
            list(GET _stamp_lines 1 _cached_supported)
            set(_reuse TRUE)
        endif()
    endif()

    if(NOT _reuse)
        if(WITH_OMPI)
            include(CheckCXXSourceCompiles)
            include(CMakePushCheckState)
            cmake_push_check_state(RESET)

            set(CMAKE_REQUIRED_INCLUDES ${_probe_includes})

            set(CMAKE_REQUIRED_LIBRARIES ${_probe_libraries})
            if(_probe_compile_flags)
                set(CMAKE_REQUIRED_FLAGS ${_probe_compile_flags})
            endif()
            if(_probe_link_flags)
                separate_arguments(_MPI_DEVICE_AWARE_LINK_OPTIONS NATIVE_COMMAND
                    "${_probe_link_flags}")
                set(CMAKE_REQUIRED_LINK_OPTIONS
                    ${_MPI_DEVICE_AWARE_LINK_OPTIONS})
            endif()

            unset(_INFINICCL_MPI_HAS_OMPI_CUDA_QUERY CACHE)
            unset(_INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
            check_cxx_source_compiles("
                #include <mpi.h>
                #include <mpi-ext.h>
                int main() {
                    return MPIX_Query_cuda_support();
                }
            " _INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
            cmake_pop_check_state()

            if(_INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
                set(_supported 1)
            endif()
        endif()

        file(WRITE "${_stamp}" "${_fingerprint_md5}\n${_supported}")
    else()
        set(_supported "${_cached_supported}")
    endif()

    set(_output
        "${CMAKE_BINARY_DIR}/src/backends/mpi/ompi/device_support_config.h")
    set(_extension_includes "")
    set(_query_body "return false;")
    if(_supported)
        set(_extension_includes
            "#include <mpi.h>\n#include <mpi-ext.h>\n")
        set(_query_body "return MPIX_Query_cuda_support() != 0;")
    endif()

    string(CONCAT _content
        "#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_DEVICE_SUPPORT_CONFIG_H_\n"
        "#define INFINI_CCL_BACKENDS_MPI_OMPI_DEVICE_SUPPORT_CONFIG_H_\n"
        "\n"
        "// Generated by `infiniccl_configure_mpi_device_aware()` at\n"
        "// configure time from a compile-link probe against the selected MPI.\n"
        "// Do not edit.\n"
        "\n"
        "${_extension_includes}"
        "\n"
        "namespace infini::ccl::mpi_config {\n"
        "\n"
        "inline constexpr bool kHasOmpiCudaQuery = ${_supported};\n"
        "\n"
        "inline bool QueryOmpiCudaSupport() {\n"
        "  ${_query_body}\n"
        "}\n"
        "\n"
        "}  // namespace infini::ccl::mpi_config\n"
        "\n"
        "#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_DEVICE_SUPPORT_CONFIG_H_\n")

    set(_header_changed TRUE)
    if(EXISTS "${_output}")
        file(READ "${_output}" _existing_content)
        if(_existing_content STREQUAL "${_content}")
            set(_header_changed FALSE)
        endif()
    endif()
    if(_header_changed)
        file(WRITE "${_output}" "${_content}")
        message(STATUS
            "MPI device-aware query: wrote ${_output}")
    else()
        message(STATUS
            "MPI device-aware query: ${_output} unchanged")
    endif()

    if(_supported)
        set(${output} ON PARENT_SCOPE)
    else()
        set(${output} OFF PARENT_SCOPE)
    endif()
endfunction()
