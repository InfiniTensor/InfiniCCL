function(infiniccl_normalize_musa_architectures input output)
    set(_architectures "${input}")
    string(REPLACE "," ";" _architectures "${_architectures}")
    string(REGEX REPLACE "[ \t\r\n]+" ";" _architectures "${_architectures}")

    set(_normalized)
    foreach(_architecture IN LISTS _architectures)
        if(_architecture STREQUAL "")
            continue()
        endif()

        string(REGEX REPLACE "^mp_" "" _architecture "${_architecture}")
        string(REPLACE "." "" _architecture "${_architecture}")
        if(NOT _architecture MATCHES "^[0-9]+$")
            message(FATAL_ERROR
                "Invalid MUSA architecture `${_architecture}`. "
                "Use values such as `31`, `mp_31`, or `3.1`."
            )
        endif()

        list(APPEND _normalized "${_architecture}")
    endforeach()

    list(REMOVE_DUPLICATES _normalized)
    if(NOT _normalized)
        message(FATAL_ERROR "At least one MUSA architecture is required.")
    endif()

    set(${output} "${_normalized}" PARENT_SCOPE)
endfunction()

function(infiniccl_compute_musa_architecture_config input architectures_output march_type_output flags_output)
    infiniccl_normalize_musa_architectures("${input}" _architectures)

    # `mccl.h` exposes data types globally from one `MARCH_TYPE`. Use the least
    # capable target so a fat binary never advertises a type that one of its
    # target architectures cannot execute.
    set(_march_type)
    set(_flags)
    foreach(_architecture IN LISTS _architectures)
        math(EXPR _architecture_version "${_architecture} * 10")
        if(NOT _march_type OR _architecture_version LESS _march_type)
            set(_march_type "${_architecture_version}")
        endif()
        list(APPEND _flags "--offload-arch=mp_${_architecture}")
    endforeach()

    set(${architectures_output} "${_architectures}" PARENT_SCOPE)
    set(${march_type_output} "${_march_type}" PARENT_SCOPE)
    set(${flags_output} "${_flags}" PARENT_SCOPE)
endfunction()

function(infiniccl_configure_musa_target target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Cannot configure unknown MUSA target `${target}`.")
    endif()

    foreach(_musa_arch_compile_option IN LISTS MUSA_ARCH_COMPILE_OPTIONS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${_musa_arch_compile_option}>
        )
    endforeach()
    target_compile_definitions(${target} PRIVATE
        MARCH_TYPE=${MUSA_MARCH_TYPE}
    )
endfunction()

function(infiniccl_detect_musa_architectures musa_include_dir musart_library output)
    set(_source "${CMAKE_CURRENT_BINARY_DIR}/get_musa_compute_capabilities.cpp")
    file(WRITE "${_source}" [=[
#include <musa_runtime.h>

#include <cstdio>

int main() {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count == 0) {
    return 1;
  }

  for (int device = 0; device < device_count; ++device) {
    musaDeviceProp properties;
    if (musaGetDeviceProperties(&properties, device) != musaSuccess) {
      return 1;
    }
    std::printf("%d%d ", properties.major, properties.minor);
  }
  return 0;
}
]=])

    try_run(
        _run_result
        _compile_result
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${_source}"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${musa_include_dir}"
        LINK_LIBRARIES "${musart_library}"
        RUN_OUTPUT_VARIABLE _detected_architectures
    )

    if(NOT _compile_result OR NOT "${_run_result}" STREQUAL "0")
        message(FATAL_ERROR
            "Could not detect the architecture of the installed MUSA GPUs. "
            "Set `MUSA_ARCHITECTURES` explicitly, for example "
            "`-DMUSA_ARCHITECTURES=31`."
        )
    endif()

    infiniccl_normalize_musa_architectures(
        "${_detected_architectures}"
        _detected_architectures
    )
    set(${output} "${_detected_architectures}" PARENT_SCOPE)
endfunction()
