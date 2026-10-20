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
file(GLOB_RECURSE BRIDGE_DEP_HEADERS CONFIGURE_DEPENDS
    "base/*.h"
    "backends/*.h"
    "devices/*.h"
    "devices/*.cuh"
)

string(REPLACE ";" "\n" BRIDGE_DEPENDENCY_CONTENT "${BRIDGE_DEP_HEADERS}")
set(BRIDGE_DEPENDENCY_MANIFEST
    "${CMAKE_CURRENT_BINARY_DIR}/bridge_dependencies.txt"
)
file(GENERATE
    OUTPUT "${BRIDGE_DEPENDENCY_MANIFEST}"
    CONTENT "${BRIDGE_DEPENDENCY_CONTENT}\n"
)

target_sources(infiniccl PRIVATE
    ${CORE_SRCS}
    ${BASE_IMPL_SRCS}
)

# =========================================================
# --- DEVICE SECTIONS (Hardware Runtimes) ---
# =========================================================

# CPU (Always On as the base architecture)
file(GLOB_RECURSE CPU_SRCS "devices/cpu/*.cc")
list(FILTER CPU_SRCS EXCLUDE REGEX "rdma_memory\\.cc$")
target_sources(infiniccl PRIVATE ${CPU_SRCS})

# NVIDIA
if(WITH_NVIDIA)
    list(APPEND DEVICE_LIST "nvidia")

    set(NVIDIA_PATTERNS
        "devices/cuda/*.cc"
        "devices/cuda/*.cpp"
        "devices/cuda/*.cu"
        "devices/nvidia/*.cc"
        "devices/nvidia/*.cpp"
        "devices/nvidia/*.cu"
    )
    file(GLOB NVIDIA_SOURCES ${NVIDIA_PATTERNS})
    list(FILTER NVIDIA_SOURCES EXCLUDE REGEX "rdma_memory\\.cc$")

    target_sources(infiniccl PRIVATE ${NVIDIA_SOURCES})
    target_link_libraries(infiniccl PRIVATE CUDA::cudart CUDA::cuda_driver)

    set_target_properties(infiniccl PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# T-Head
if(WITH_THEAD)
    list(APPEND DEVICE_LIST "thead")

    set(THEAD_PATTERNS
        "devices/cuda/*.cc"
        "devices/cuda/*.cpp"
        "devices/cuda/*.cu"
        "devices/thead/*.cc"
        "devices/thead/*.cpp"
        "devices/thead/*.cu"
    )
    file(GLOB THEAD_SOURCES ${THEAD_PATTERNS})

    target_sources(infiniccl PRIVATE ${THEAD_SOURCES})
    target_link_libraries(infiniccl PRIVATE CUDA::cudart CUDA::cuda_driver)

    set_target_properties(infiniccl PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# Iluvatar
if(WITH_ILUVATAR)
    list(APPEND DEVICE_LIST "iluvatar")

    set(ILUVATAR_PATTERNS
        "devices/cuda/*.cc"
        "devices/cuda/*.cpp"
        "devices/cuda/*.cu"
        "devices/iluvatar/*.cc"
        "devices/iluvatar/*.cpp"
        "devices/iluvatar/*.cu"
    )
    file(GLOB_RECURSE ILUVATAR_SOURCES ${ILUVATAR_PATTERNS})
    target_sources(infiniccl PRIVATE ${ILUVATAR_SOURCES})

    target_include_directories(infiniccl PRIVATE "${ILUVATAR_INCLUDE_DIR}")

    set_source_files_properties(${ILUVATAR_SOURCES} PROPERTIES
        COMPILE_OPTIONS "${ILUVATAR_CUDA_FLAGS}"
    )
    set(CMAKE_CXX_COMPILER "${ILUVATAR_CUDA_COMPILER}" CACHE FILEPATH "Iluvatar Compiler" FORCE)
    target_compile_options(infiniccl PRIVATE ${ILUVATAR_CUDA_FLAGS})

    target_link_libraries(infiniccl PRIVATE CUDA::cudart)
endif()

# MetaX
if(WITH_METAX)
    list(APPEND DEVICE_LIST "metax")

    set(METAX_PATTERNS
        "devices/cuda/*.cc"
        "devices/cuda/*.cpp"
        "devices/metax/*.cc"
        "devices/metax/*.cpp"
        "devices/metax/*.maca"
    )
    file(GLOB METAX_SOURCES ${METAX_PATTERNS})
    list(FILTER METAX_SOURCES EXCLUDE REGEX "rdma_memory\\.cc$")

    set_source_files_properties(${METAX_SOURCES} PROPERTIES LANGUAGE CXX)

    target_sources(infiniccl PRIVATE ${METAX_SOURCES})
    target_include_directories(infiniccl PRIVATE "${MACA_PATH}/include")
    target_link_libraries(infiniccl PRIVATE ${MACA_RUNTIME_LIB})
    target_compile_options(infiniccl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x maca>)
endif()

# Mthreads (Moore)
if(WITH_MOORE)
    list(APPEND DEVICE_LIST "moore")

    set(MOORE_PATTERNS
        "devices/cuda/*.cc"
        "devices/cuda/*.cpp"
        "devices/moore/*.cc"
        "devices/moore/*.cpp"
        "devices/moore/*.mu"
    )
    file(GLOB MOORE_SOURCES ${MOORE_PATTERNS})

    set_source_files_properties(${MOORE_SOURCES} PROPERTIES LANGUAGE CXX)

    # target_compile_options(infiniccl PRIVATE "-x" "musa")
    target_sources(infiniccl PRIVATE ${MOORE_SOURCES})

    target_include_directories(infiniccl PRIVATE "${MUSA_ROOT}/include")
    target_link_libraries(infiniccl PRIVATE ${MUSA_LIB} ${MUSART_LIB} ${MUBLAS_LIB})
    target_compile_options(infiniccl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x musa>)
    infiniccl_configure_musa_target(infiniccl)
endif()

# Cambricon
if(WITH_CAMBRICON)
    list(APPEND DEVICE_LIST "cambricon")

    set(CAMBRICON_PATTERNS
        "devices/cambricon/*.cc"
        "devices/cambricon/*.cpp"
        "devices/cambricon/*.mlu"
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

# Ascend
if(WITH_ASCEND)
    list(APPEND DEVICE_LIST "ascend")

    file(GLOB_RECURSE ASCEND_SRCS "devices/ascend/*.cc" "devices/ascend/*.cpp")

    target_sources(infiniccl PRIVATE ${ASCEND_SRCS})
    target_include_directories(infiniccl PRIVATE ${ASCEND_INC} ${ASCEND_OPDEV_INC})
    target_link_libraries(infiniccl PRIVATE ${ASCENDCL_LIB} ${ASCEND_NNOPBASE_LIB})
endif()

# Hygon
if(WITH_HYGON)
    list(APPEND DEVICE_LIST "hygon")

    set(HYGON_PATTERNS
        "devices/hygon/*.cc"
        "devices/hygon/*.cpp"
    )
    file(GLOB_RECURSE HYGON_SRCS ${HYGON_PATTERNS})

    set_source_files_properties(${HYGON_SRCS} PROPERTIES LANGUAGE CXX)

    target_sources(infiniccl PRIVATE ${HYGON_SRCS})
    target_include_directories(infiniccl PRIVATE ${HYGON_HIP_INCLUDE_DIR})
    target_compile_definitions(infiniccl PRIVATE ${HYGON_HIP_DEFINITIONS})
    target_link_libraries(infiniccl PRIVATE ${HYGON_HIP_RUNTIME_LIB})
endif()

# =========================================================
# --- BACKEND SECTIONS (Communication Protocols) ---
# =========================================================

# OpenMPI
if(WITH_OMPI)
    list(APPEND BACKEND_LIST "ompi")
    file(GLOB_RECURSE OMPI_SRCS "backends/mpi/ompi/*.cc" "backends/mpi/ompi/*.cpp")
    target_sources(infiniccl PRIVATE ${OMPI_SRCS})

    if(TARGET MPI::MPI_CXX)
        target_link_libraries(infiniccl PRIVATE MPI::MPI_CXX)
    endif()
endif()

# RDMA (transport attribute of the MPI backends)
if(WITH_RDMA)
    if(WITH_OMPI OR WITH_MPICH)
        # The transport exchanges its control frames and queue-pair metadata
        # over MPI, so it only builds together with an MPI backend.
        target_sources(infiniccl PRIVATE
            backends/mpi/rdma/rdma_transport.cc
            backends/mpi/rdma/memory_adaptor.cc
            devices/cpu/rdma_memory.cc
        )
        target_include_directories(infiniccl PRIVATE
            ${IBVERBS_INCLUDE_DIR}
        )
        target_link_libraries(infiniccl PRIVATE
            ${IBVERBS_LIBRARY}
            Threads::Threads
        )
        if(TARGET MPI::MPI_CXX)
            target_link_libraries(infiniccl PRIVATE MPI::MPI_CXX)
        endif()
        target_compile_definitions(infiniccl PRIVATE INFINICCL_WITH_RDMA)
        find_path(MLX5DV_INCLUDE_DIR NAMES infiniband/mlx5dv.h)
        find_library(MLX5DV_LIBRARY NAMES mlx5)
        if(MLX5DV_INCLUDE_DIR AND MLX5DV_LIBRARY)
            include(CMakePushCheckState)
            include(CheckCXXSourceCompiles)
            cmake_push_check_state(RESET)
            set(CMAKE_REQUIRED_INCLUDES ${MLX5DV_INCLUDE_DIR} ${IBVERBS_INCLUDE_DIR})
            set(CMAKE_REQUIRED_LIBRARIES ${MLX5DV_LIBRARY} ${IBVERBS_LIBRARY})
            check_cxx_source_compiles("
                #include <infiniband/mlx5dv.h>
                int main() {
                    ibv_qp_init_attr_ex attr{};
                    mlx5dv_qp_init_attr mlx5_attr{};
                    attr.comp_mask = IBV_QP_INIT_ATTR_PD;
                    mlx5_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
                    mlx5_attr.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
                    auto *qp = mlx5dv_create_qp(nullptr, &attr, &mlx5_attr);
                    return mlx5dv_is_supported(nullptr) && qp ? 0 : 1;
                }
            " INFINICCL_HAS_MLX5DV_SCATTER_CONTROL)
            cmake_pop_check_state()
        endif()
        if(MLX5DV_INCLUDE_DIR AND MLX5DV_LIBRARY AND
           INFINICCL_HAS_MLX5DV_SCATTER_CONTROL)
            target_include_directories(infiniccl PRIVATE ${MLX5DV_INCLUDE_DIR})
            target_link_libraries(infiniccl PRIVATE ${MLX5DV_LIBRARY})
            target_compile_definitions(infiniccl PRIVATE INFINICCL_WITH_MLX5DV)
            message(STATUS "RDMA: per-QP mlx5 Scatter-to-CQE disabling enabled.")
        else()
            message(STATUS "RDMA: mlx5dv unavailable; Mellanox QPs will be rejected for safety.")
        endif()
        if(WITH_NVIDIA)
            if(CUDAToolkit_VERSION VERSION_LESS 11.3)
                message(FATAL_ERROR "NVIDIA RDMA requires CUDA 11.3 or newer for visibility capability checks.")
            endif()
            target_sources(infiniccl PRIVATE devices/nvidia/rdma_memory.cc)
            target_compile_definitions(infiniccl PRIVATE INFINICCL_WITH_NVIDIA_RDMA)
        endif()
        if(WITH_METAX)
            find_path(METAX_RDMA_INCLUDE_DIR NAMES mxkw/mxkw.h
                HINTS "${MACA_PATH}/include" REQUIRED)
            find_library(METAX_RDMA_LIBRARY NAMES mxkw
                HINTS "${MACA_PATH}/lib" "${MACA_PATH}/lib64" REQUIRED)
            target_sources(infiniccl PRIVATE devices/metax/rdma_memory.cc)
            target_include_directories(infiniccl PRIVATE ${METAX_RDMA_INCLUDE_DIR})
            target_link_libraries(infiniccl PRIVATE ${METAX_RDMA_LIBRARY})
            target_compile_definitions(infiniccl PRIVATE INFINICCL_WITH_METAX_RDMA)
        endif()
    else()
        set(WITH_RDMA OFF PARENT_SCOPE)
        message(STATUS "RDMA: the transport requires an MPI backend (WITH_OMPI or WITH_MPICH); it is disabled.")
    endif()
endif()

if(WITH_MPICH)
    list(APPEND BACKEND_LIST "mpich")
    # Note: MPICH shares the same internal implementation as OMPI, so we reuse the `ompi/impl` sources for both.
    # However, later there may be MPICH-specific sources added to the `mpich/` directory.
    file(GLOB_RECURSE MPICH_SRCS "backends/mpi/mpich/*.cc" "backends/mpi/mpich/*.cpp" "backends/mpi/ompi/*.cc" "backends/mpi/ompi/*.cpp")
    target_sources(infiniccl PRIVATE ${MPICH_SRCS})
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(infiniccl PRIVATE MPI::MPI_CXX)
    endif()
endif()

# NCCL
if(WITH_NCCL)
    list(APPEND BACKEND_LIST "nccl")
    file(GLOB_RECURSE NCCL_SRCS "backends/ccl/nccl/*.cc" "backends/ccl/nccl/*.cpp" "backends/ccl/nccl/*.cu")

    target_sources(infiniccl PRIVATE ${NCCL_SRCS})
    target_include_directories(infiniccl PRIVATE ${NCCL_INC})
    target_link_libraries(infiniccl PRIVATE ${NCCL_LIB})

    if(NCCL_COMPILE_DEFINITIONS)
        target_compile_definitions(infiniccl PRIVATE ${NCCL_COMPILE_DEFINITIONS})
    endif()
endif()

# MCCL
if(WITH_MCCL)
    list(APPEND BACKEND_LIST "mccl")
    file(GLOB_RECURSE MCCL_SRCS "backends/ccl/mccl/*.cc" "backends/ccl/mccl/*.cpp" "backends/ccl/mccl/*.maca")

    target_sources(infiniccl PRIVATE ${MCCL_SRCS})
    target_include_directories(infiniccl PRIVATE ${MCCL_INC})
    target_link_libraries(infiniccl PRIVATE ${MCCL_LIB})
endif()

# CNCL
if(WITH_CNCL)
    list(APPEND BACKEND_LIST "cncl")
    file(GLOB_RECURSE CNCL_SRCS "backends/ccl/cncl/*.cc" "backends/ccl/cncl/*.cpp")

    target_sources(infiniccl PRIVATE ${CNCL_SRCS})
    target_include_directories(infiniccl PRIVATE ${CNCL_INC})
    target_link_libraries(infiniccl PRIVATE ${CNCL_LIB})
endif()

# HCCL
if(WITH_HCCL)
    list(APPEND BACKEND_LIST "hccl")
    file(GLOB_RECURSE HCCL_SRCS "backends/ccl/hccl/*.cc" "backends/ccl/hccl/*.cpp")
    
    target_sources(infiniccl PRIVATE ${HCCL_SRCS})
    target_include_directories(infiniccl PRIVATE ${HCCL_INC} ${HCCL_ACLNN_INC})
    target_link_libraries(infiniccl PRIVATE ${HCCL_LIB} ${HCCL_OPAPI_LIB})
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

set(INFINICCL_H_CONTENT "${INFINICCL_H_CONTENT}\n#endif  // INFINI_CCL_H_\n")

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
            "${BRIDGE_DEPENDENCY_MANIFEST}"
            ${BASE_IMPL_SRCS}
            ${BRIDGE_DEP_HEADERS}
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
