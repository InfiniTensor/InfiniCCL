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
