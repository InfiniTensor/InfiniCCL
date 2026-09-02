include("${CMAKE_CURRENT_LIST_DIR}/../MooreArchitecture.cmake")

if(TEST_CASE STREQUAL "invalid")
    infiniccl_normalize_musa_architectures("31;not-an-architecture" _unused)
    message(FATAL_ERROR "Invalid architecture input was accepted")
endif()

infiniccl_compute_musa_architecture_config(
    "31, mp_22 3.1;21"
    _architectures
    _march_type
    _flags
)

if(NOT _architectures STREQUAL "31;22;21")
    message(FATAL_ERROR "Unexpected normalized architectures: ${_architectures}")
endif()
if(NOT _march_type EQUAL 210)
    message(FATAL_ERROR "MARCH_TYPE must represent the least capable target: ${_march_type}")
endif()
if(NOT _flags STREQUAL "--offload-arch=mp_31;--offload-arch=mp_22;--offload-arch=mp_21")
    message(FATAL_ERROR "Unexpected MUSA compiler flags: ${_flags}")
endif()
