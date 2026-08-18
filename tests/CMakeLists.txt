add_test(
    NAME bridge_dependency_contract
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/bridge_dependency_contract.py"
            "${PROJECT_SOURCE_DIR}/src/CMakeLists.txt"
)

add_test(
    NAME bridge_generation
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/bridge_generation.py"
            "${PROJECT_SOURCE_DIR}"
)

add_executable(ccl_point_to_point_impl_test
    ccl_point_to_point_impl.cc
)

target_include_directories(ccl_point_to_point_impl_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

add_test(
    NAME ccl_point_to_point_impl
    COMMAND ccl_point_to_point_impl_test
)
