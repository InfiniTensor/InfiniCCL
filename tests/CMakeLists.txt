add_test(
    NAME bridge_dependency_contract
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/bridge_dependency_contract.py"
            "${PROJECT_SOURCE_DIR}/src/CMakeLists.txt"
)
