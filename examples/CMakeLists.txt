file(GLOB_RECURSE EXAMPLE_SOURCES "*.cc")

foreach(source_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)

    add_executable(${example_name} ${source_file})

    target_link_libraries(${example_name} PRIVATE infiniccl)

    target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
