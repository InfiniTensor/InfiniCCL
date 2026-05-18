add_library(infiniccl SHARED)

set(DEVICE_LIST "cpu")
set(BACKEND_LIST "")
set(AUTOGEN_WARNING [[/*
 * -----------------------------------------------------------------
 * AUTO-GENERATED FILE — DO NOT EDIT.
 * -----------------------------------------------------------------
 * This file is generated during the build process.
 * Any changes made here may be overwritten during the build or code
 * generation process.
 * -----------------------------------------------------------------
 */

]])

file(GLOB CORE_SRCS "*.cc")
file(GLOB_RECURSE BASE_IMPL_SRCS "base/*.cc")

target_sources(infiniccl PRIVATE 
    ${CORE_SRCS} 
    ${BASE_IMPL_SRCS}
)

# =========================================================
# --- DEVICE SECTIONS (Hardware Runtimes) ---
# =========================================================

# CPU (Always On as the base architecture)
file(GLOB_RECURSE CPU_SRCS "cpu/*.cc")
target_sources(infiniccl PRIVATE ${CPU_SRCS})

# NVIDIA
if(WITH_NVIDIA)
    list(APPEND DEVICE_LIST "nvidia")

    set(NVIDIA_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "cuda/*.cu"
        "nvidia/*.cc"
        "nvidia/*.cpp"
        "nvidia/*.cu"
    )
    file(GLOB NVIDIA_SOURCES ${NVIDIA_PATTERNS})

    target_sources(infiniccl PRIVATE ${NVIDIA_SOURCES})
    target_link_libraries(infiniccl PRIVATE CUDA::cudart CUDA::cuda_driver)

    set_target_properties(infiniccl PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# MetaX
if(WITH_METAX)
    list(APPEND DEVICE_LIST "metax")

    set(METAX_PATTERNS
        "cuda/*.cc"
        "cuda/*.cpp"
        "metax/*.cc"
        "metax/*.cpp"
        "metax/*.maca"
    )
    file(GLOB METAX_SOURCES ${METAX_PATTERNS})

    set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)
    
    target_sources(infiniccl PRIVATE ${METAX_SOURCES})
    target_include_directories(infiniccl PRIVATE "${MACA_PATH}/include")
    target_link_libraries(infiniccl PRIVATE ${MACA_RUNTIME_LIB})
    target_compile_options(infiniccl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x maca>)
endif()

# Cambricon
if(WITH_CAMBRICON)
    list(APPEND DEVICE_LIST "cambricon")

    set(CAMBRICON_PATTERNS
        "cambricon/*.cc"
        "cambricon/*.cpp"
        "cambricon/*.mlu"
    )
    file(GLOB CAMBRICON_SOURCES ${CAMBRICON_PATTERNS})

    find_program(CNCC_COMPILER cncc HINTS "${NEUWARE_HOME}/bin" "$ENV{NEUWARE_HOME}/bin" /usr/local/neuware/bin)

    if(CNCC_COMPILER)
        message(STATUS "Found Cambricon `cncc` compiler: ${CNCC_COMPILER}")
        execute_process(
            COMMAND ${CNCC_COMPILER} --print-file-name=include
            OUTPUT_VARIABLE CNCC_INTERNAL_INC_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        set(MLU_COMPILE_OPTS
            -c --bang-mlu-arch=mtp_592 -O3 -fPIC -Wall -Werror -std=c++17 -pthread
            -I${PROJECT_SOURCE_DIR}
            -I${PROJECT_SOURCE_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${NEUWARE_HOME}/include
            -I${CNCC_INTERNAL_INC_DIR}
        )

        # Custom automation function to route `.mlu` sources through `cncc`.
        function(compile_mlu_file src_file)
            get_filename_component(name ${src_file} NAME_WE)
            get_filename_component(path ${src_file} DIRECTORY)
            set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${path}/${name}.o")
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${path}")
            
            add_custom_command(OUTPUT ${out_file}
                COMMAND ${CNCC_COMPILER} ${MLU_COMPILE_OPTS} -c ${src_file} -o ${out_file}
                DEPENDS ${src_file}
                COMMENT "Building MLU kernel: ${src_file}"
            )
            set_property(DIRECTORY APPEND PROPERTY CAMBRICON_OBJECTS ${out_file})
        endfunction()

        foreach(src ${CAMBRICON_SOURCES})
            if(${src} MATCHES "\\.mlu$")
                compile_mlu_file(${src})
            endif()
        endforeach()

        # Append generated MLU hardware binaries straight into the target.
        get_directory_property(CAMBRICON_OBJECT_FILES CAMBRICON_OBJECTS)
        if(CAMBRICON_OBJECT_FILES)
            target_sources(infiniccl PRIVATE ${CAMBRICON_OBJECT_FILES})
        endif()
    else()
        message(FATAL_ERROR "`cncc` compiler required for Cambricon build but not found.")
    endif()

    foreach(src ${CAMBRICON_SOURCES})
        if(${src} MATCHES "\\.(cc|cpp)$")
            target_sources(infiniccl PRIVATE ${src})
        endif()
    endforeach()

    target_include_directories(infiniccl PRIVATE "${NEUWARE_HOME}/include")
    
    target_link_libraries(infiniccl PRIVATE 
        ${CAMBRICON_RUNTIME_LIB} 
        ${CAMBRICON_CNNL_LIB} 
        ${CAMBRICON_CNNL_EXTRA_LIB} 
        ${CAMBRICON_PAPI_LIB}
    )
endif()

# =========================================================
# --- BACKEND SECTIONS (Communication Protocols) ---
# =========================================================

# OpenMPI
if(WITH_OMPI)
    list(APPEND BACKEND_LIST "ompi")
    file(GLOB_RECURSE OMPI_SRCS "ompi/*.cc" "ompi/*.cpp")
    target_sources(infiniccl PRIVATE ${OMPI_SRCS})
    
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(infiniccl PRIVATE MPI::MPI_CXX)
    endif()
endif()

if(WITH_MPICH)
    list(APPEND BACKEND_LIST "mpich")
    # Note: MPICH shares the same internal implementation as OMPI, so we reuse the `ompi/impl` sources for both.
    # However, later there may be MPICH-specific sources added to the `mpich/` directory.
    file(GLOB_RECURSE MPICH_SRCS "mpich/*.cc" "mpich/*.cpp" "ompi/*.cc" "ompi/*.cpp") 
    target_sources(infiniccl PRIVATE ${MPICH_SRCS})
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(infiniccl PRIVATE MPI::MPI_CXX)
    endif()
endif()

# NCCL
if(WITH_NCCL)
    list(APPEND BACKEND_LIST "nccl")

    if (WITH_NVIDIA)
        file(GLOB_RECURSE NCCL_SRCS "nvidia/nccl/*.cc" "nvidia/nccl/*.cpp" "nvidia/nccl/*.cu")
    endif()

    target_sources(infiniccl PRIVATE ${NCCL_SRCS})
    target_include_directories(infiniccl PRIVATE ${NCCL_INC})
    target_link_libraries(infiniccl PRIVATE ${NCCL_LIB})
endif()

# =========================================================
# --- File Generation ---
# =========================================================
# --- Generate umbrella header: `include/infiniccl.h` ---
set(PUBLIC_HEADERS_DIR "${PROJECT_SOURCE_DIR}/include")
file(GLOB PUBLIC_HEADERS "${PUBLIC_HEADERS_DIR}/*.h")

set(INFINICCL_H_CONTENT "${AUTOGEN_WARNING}#ifndef INFINI_CCL_H_\n#define INFINI_CCL_H_\n\n")

foreach(HEADER_PATH ${PUBLIC_HEADERS})
    get_filename_component(HEADER_NAME ${HEADER_PATH} NAME)
    if(NOT "${HEADER_NAME}" STREQUAL "infiniccl.h")
        set(INFINICCL_H_CONTENT "${INFINICCL_H_CONTENT}#include \"${HEADER_NAME}\"\n")
    endif()
endforeach()

set(INFINICCL_H_CONTENT "${INFINICCL_H_CONTENT}\n#endif // INFINI_CCL_H_\n")

file(WRITE "${PUBLIC_HEADERS_DIR}/infiniccl.h" "${INFINICCL_H_CONTENT}")

# --- Generate the bridge files ---
set(GENERATED_BRIDGE "${CMAKE_CURRENT_BINARY_DIR}/comm_bridge.cc")
set(GENERATED_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/backend_manifest.h")
set(DEV_STR "${DEVICE_LIST}")
set(BACK_STR "${BACKEND_LIST}")

add_custom_command(
    OUTPUT "${GENERATED_BRIDGE}" "${GENERATED_MANIFEST}"
    COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/gen_bridge.py" 
            "${PROJECT_SOURCE_DIR}" 
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${DEV_STR}"
            "${BACK_STR}"
    DEPENDS "${PROJECT_SOURCE_DIR}/scripts/gen_bridge.py"
            "${PROJECT_SOURCE_DIR}/include/comm.h"
            ${BASE_IMPL_SRCS}
    VERBATIM
    COMMENT "Generating InfiniCCL bridge and manifest files for Devices: [${DEV_STR}] Backends: [${BACK_STR}]..."
)

target_sources(infiniccl PRIVATE "${GENERATED_BRIDGE}")
target_include_directories(infiniccl PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# =========================================================
# --- Interface and Export ---
# =========================================================

target_include_directories(infiniccl
    PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

string(REPLACE ";" ", " PRINT_DEVS "${DEVICE_LIST}")
string(REPLACE ";" ", " PRINT_BACKS "${BACKEND_LIST}")
message(STATUS "InfiniCCL Config: Devices [${PRINT_DEVS}] | Backends [${PRINT_BACKS}]")
