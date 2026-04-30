file(GLOB_RECURSE EXAMPLE_SOURCES "*.cc")

foreach(source_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)

    add_executable(${example_name} ${source_file})

    target_link_libraries(${example_name} PRIVATE infiniccl)

    # Add runtime and backend dependencies for direct runtime/backend usage.
    if(WITH_NVIDIA)
        target_link_libraries(${example_name} PRIVATE CUDA::cudart)
    endif()
    
    if(WITH_OMPI)
        target_link_libraries(${example_name} PRIVATE MPI::MPI_CXX)
    endif()

    # Explicitly allow examples to "peek" into the internal `src` and binary dirs.
    # This is necessary because these were marked `PRIVATE` in the library's CMake.
    target_include_directories(${example_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${PROJECT_SOURCE_DIR}/src"         # For internal templates like `runtime.h`.
        "${CMAKE_BINARY_DIR}/src"           # For the generated `backend_manifest.h`.
    )
endforeach()
