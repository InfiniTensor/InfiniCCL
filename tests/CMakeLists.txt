if(NOT WITH_MCCL)
  return()
endif()

add_executable(ccl_collectives_impl_test
    ccl_collectives_impl.cc
)

target_include_directories(ccl_collectives_impl_test PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}/src"
)
target_compile_features(ccl_collectives_impl_test PRIVATE cxx_std_17)

add_test(
    NAME ccl_collectives_impl
    COMMAND ccl_collectives_impl_test
)
