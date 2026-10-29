cmake_minimum_required(VERSION 3.18)
project(InfiniCCL VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# --- DEVICE OPTIONS (Hardware Runtimes) ---
# =========================================================
option(WITH_NVIDIA      "Enable NVIDIA GPU support"     OFF)
option(WITH_ILUVATAR    "Enable ILUVATAR GPU support"   OFF)
option(WITH_METAX       "Enable MetaX GPU support"      OFF)
option(WITH_MOORE       "Enable Moore GPU support"      OFF)
option(WITH_CAMBRICON   "Enable Cambricon MLU support"  OFF)
option(WITH_ASCEND      "Enable Ascend NPU support"     OFF)
option(WITH_HYGON       "Enable Hygon DCU support"      OFF)
option(WITH_THEAD       "Enable T-Head PPU support"     OFF)

set(WITH_CPU ON CACHE INTERNAL "CPU backend is always enabled")

# =========================================================
# --- BACKEND OPTIONS (Communication Protocols) ---
# =========================================================
option(WITH_OMPI "Enable OpenMPI backend" OFF)
option(WITH_MPICH "Enable MPICH backend" OFF)
option(WITH_NCCL "Enable NCCL backend"    OFF)
option(WITH_MCCL "Enable MCCL backend"    OFF)
option(WITH_CNCL "Enable CNCL backend"    OFF)
option(WITH_HCCL "Enable HCCL backend"    OFF)

# =========================================================
# --- MISC. BUILD OPTIONS ---
# =========================================================
option(AUTO_DETECT_DEVICES "Automatically detect available devices" ON)
option(AUTO_DETECT_BACKENDS "Automatically detect available backends" OFF)
option(BUILD_EXAMPLES "Build internal examples" ON)

# Device modules provide platform probes and SDK configuration helpers.
include(cmake/devices/DeviceDetection.cmake)
include(cmake/devices/Nvidia.cmake)
include(cmake/devices/Iluvatar.cmake)
include(cmake/devices/MetaX.cmake)
include(cmake/devices/Moore.cmake)
include(cmake/devices/Cambricon.cmake)
include(cmake/devices/Ascend.cmake)
include(cmake/devices/Hygon.cmake)
include(cmake/devices/THead.cmake)

# Backend modules provide backend probes and required dependency resolution.
include(cmake/backends/MPI.cmake)
include(cmake/backends/NCCL.cmake)
include(cmake/backends/MCCL.cmake)
include(cmake/backends/CNCL.cmake)
include(cmake/backends/HCCL.cmake)

# =========================================================
# --- AUTO-DETECTION: DEVICES ---
# =========================================================
if(AUTO_DETECT_DEVICES)
    message(STATUS "Auto-detecting available devices...")

    # NVIDIA
    set(NVIDIA_REQUESTED ${WITH_NVIDIA})
    infiniccl_resolve_device_detection(
        WITH_NVIDIA "NVIDIA" infiniccl_detect_nvidia NVIDIA_FOUND)

    # Iluvatar
    infiniccl_resolve_device_detection(
        WITH_ILUVATAR "Iluvatar" infiniccl_detect_iluvatar ILUVATAR_FOUND)

    # MetaX
    infiniccl_resolve_device_detection(
        WITH_METAX "MetaX" infiniccl_detect_metax METAX_FOUND)

    # MThreads (Moore)
    infiniccl_resolve_device_detection(
        WITH_MOORE "Moore" infiniccl_detect_moore MOORE_FOUND)

    # Cambricon
    infiniccl_resolve_device_detection(
        WITH_CAMBRICON "Cambricon" infiniccl_detect_cambricon CAMBRICON_FOUND)

    # Ascend
    infiniccl_resolve_device_detection(
        WITH_ASCEND "Ascend" infiniccl_detect_ascend ASCEND_FOUND)

    # HYGON
    infiniccl_resolve_device_detection(
        WITH_HYGON "Hygon" infiniccl_detect_hygon HYGON_FOUND)

    # T-Head
    infiniccl_resolve_device_detection(
        WITH_THEAD "T-Head" infiniccl_detect_thead THEAD_FOUND)

    # NVIDIA Final Resolution: resolve conflicts.
    if(THEAD_FOUND AND NVIDIA_FOUND AND NOT NVIDIA_REQUESTED)
        set(NVIDIA_FOUND FALSE)
        set(WITH_NVIDIA OFF CACHE BOOL "Enable NVIDIA GPU support" FORCE)
        message(STATUS "T-Head environment detected; ignoring NVIDIA auto-detection.")
    endif()
endif()

# =========================================================
# --- AUTO-DETECTION: BACKENDS ---
# =========================================================
if(AUTO_DETECT_BACKENDS)
    message(STATUS "Auto-detecting available communication backends...")

    # Detect MPI Dependencies
    infiniccl_detect_mpi(MPI_IMPLEMENTATION)
    if(MPI_IMPLEMENTATION STREQUAL "OMPI")
        set(WITH_OMPI ON)
        message(STATUS "Auto-detected OpenMPI environment.")
    elseif(MPI_IMPLEMENTATION STREQUAL "MPICH")
        set(WITH_MPICH ON)
        message(STATUS "Auto-detected MPICH/Hydra environment.")
    endif()

    # Detect NCCL Dependencies
    if(WITH_NVIDIA OR WITH_THEAD OR WITH_ILUVATAR OR WITH_HYGON)
        infiniccl_probe_nccl(NCCL_FOUND)
        if(NCCL_FOUND)
            set(WITH_NCCL ON)
            message(STATUS "Auto-detected NCCL backend.")
        else()
            message(STATUS "NCCL-compatible library/headers not found in configured paths.")
        endif()
    else()
        message(STATUS "No suitable device environment, skipping NCCL detection.")
    endif()

    # Detect MCCL Dependencies
    if(WITH_METAX OR WITH_MOORE)
        infiniccl_probe_mccl(MCCL_FOUND)
        if(MCCL_FOUND)
            set(WITH_MCCL ON)
            message(STATUS "Auto-detected MCCL backend.")
        else()
            message(STATUS "MCCL library/headers not found in configured paths.")
        endif()
    else()
        message(STATUS "No suitable device environment, skipping MCCL detection.")
    endif()

    # Detect CNCL Dependencies
    if(WITH_CAMBRICON)
        infiniccl_probe_cncl(CNCL_FOUND)
        if(CNCL_FOUND)
            set(WITH_CNCL ON)
            message(STATUS "Auto-detected CNCL backend.")
        else()
            message(STATUS "CNCL library/headers not found in Cambricon paths.")
        endif()
    endif()

    # Detect HCCL Dependencies
    if(WITH_ASCEND)
        infiniccl_probe_hccl(HCCL_FOUND)
        if(HCCL_FOUND)
            set(WITH_HCCL ON)
            message(STATUS "Auto-detected HCCL backend.")
        else()
            message(STATUS "HCCL library/headers not found in Ascend paths.")
        endif()
    endif()
endif()

# Fallback: If no backends are enabled or auto-detected, fall back to OpenMPI as the default bootstrap profile.
if(NOT WITH_OMPI AND NOT WITH_MPICH AND NOT WITH_NCCL AND NOT WITH_MCCL AND NOT WITH_CNCL AND NOT WITH_HCCL)
    set(WITH_OMPI ON)
    message(STATUS "No backend specified or detected. Defaulting to `WITH_OMPI=ON`")
endif()

# =========================================================
# --- DEPENDENCY DISCOVERY ---
# =========================================================
if(WITH_THEAD)
    infiniccl_resolve_thead_cuda(THEAD_CUDA_ROOT _THEAD_CUDA_HINTS)
    set(CUDAToolkit_ROOT "${THEAD_CUDA_ROOT}"
        CACHE PATH "T-Head CUDA-compatible toolkit root" FORCE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_NVIDIA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_ILUVATAR)
    infiniccl_configure_iluvatar()
    message(STATUS
        "Iluvatar: CUDA compiler ${ILUVATAR_CUDA_COMPILER}, "
        "arch ${ILUVATAR_ARCH}, toolkit ${ILUVATAR_CUDA_ROOT}")
endif()

if(WITH_METAX)
    infiniccl_configure_metax()
endif()

if(WITH_MOORE)
    infiniccl_configure_moore()
endif()

if(WITH_CAMBRICON)
    infiniccl_configure_cambricon()
endif()

if(WITH_ASCEND)
    infiniccl_configure_ascend(ASCEND_INC ASCENDCL_LIB)
endif()

if(WITH_HYGON)
    infiniccl_configure_hygon()
endif()

if(WITH_OMPI OR WITH_MPICH)
    find_package(MPI REQUIRED)
endif()

if(WITH_NCCL)
    infiniccl_configure_nccl(NCCL_INC NCCL_LIB NCCL_COMPILE_DEFINITIONS)
    include_directories(${NCCL_INC})
endif()

if(WITH_MCCL)
    infiniccl_configure_mccl(MCCL_INC MCCL_LIB)
    include_directories(${MCCL_INC})
endif()

if(WITH_CNCL)
    infiniccl_configure_cncl(CNCL_INC CNCL_LIB)
    include_directories(${CNCL_INC})
endif()

if(WITH_HCCL)
    infiniccl_configure_hccl(HCCL_INC HCCL_LIB)
endif()

# Python is required for code generation.
find_package(Python3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =========================================================
# --- Installation ---
# =========================================================
# For Development: Copy launcher to `build/`.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/icclrun.py"
    "${CMAKE_BINARY_DIR}/icclrun"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/icclrun_logic.py"
    "${CMAKE_BINARY_DIR}/icclrun_logic.py"
    COPYONLY
)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/backends" DESTINATION "${CMAKE_BINARY_DIR}")

# Ensure it is executable in `build/`.
file(CHMOD "${CMAKE_BINARY_DIR}/icclrun" 
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# For Production/System Install
include(GNUInstallDirs)

# Install the library and record it in an export set so that downstream
# projects can pull in `InfiniCCL::infiniccl` via `find_package(InfiniCCL)`.
# All hardware/backend dependencies (CUDA, MPI, NCCL, ...) are linked PRIVATE
# into this shared library, so they are NOT propagated as usage requirements.
install(TARGETS infiniccl
    EXPORT InfiniCCLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/infiniccl)

install(PROGRAMS scripts/icclrun.py DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME icclrun)

# =========================================================
# --- CMake Package Config (`find_package(InfiniCCL)`) ---
# =========================================================
set(INFINICCL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/InfiniCCL)

# Export the imported target definition (`InfiniCCL::infiniccl`).
install(EXPORT InfiniCCLTargets
    FILE InfiniCCLTargets.cmake
    NAMESPACE InfiniCCL::
    DESTINATION ${INFINICCL_CMAKE_DIR}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/InfiniCCLConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfig.cmake"
    INSTALL_DESTINATION ${INFINICCL_CMAKE_DIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/InfiniCCLConfigVersion.cmake"
    DESTINATION ${INFINICCL_CMAKE_DIR}
)

# Install the supporting logic to a private folder.
set(ICCL_PRIVATE_LIB ${CMAKE_INSTALL_LIBDIR}/infiniccl)
install(FILES scripts/icclrun_logic.py DESTINATION ${ICCL_PRIVATE_LIB})
install(DIRECTORY scripts/backends DESTINATION ${ICCL_PRIVATE_LIB})
