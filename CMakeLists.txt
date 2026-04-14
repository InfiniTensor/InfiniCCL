cmake_minimum_required(VERSION 3.18)
project(InfiniCCL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# --- DEVICE OPTIONS (Hardware Runtimes) ---
# =========================================================
option(WITH_NVIDIA "Enable NVIDIA GPU support" OFF)
option(WITH_METAX  "Enable MetaX GPU support"  OFF)

set(WITH_CPU ON CACHE INTERNAL "CPU backend is always enabled")

# =========================================================
# --- BACKEND OPTIONS (Communication Protocols) ---
# =========================================================
option(WITH_OMPI "Enable OpenMPI backend" OFF)
option(WITH_NCCL "Enable NCCL backend"    OFF)
# OMPI is the default bootstrap/CPU backend
if(NOT WITH_OMPI AND NOT WITH_NCCL)
    set(WITH_OMPI ON)
    message(STATUS "No backend specified. Defaulting to WITH_OMPI=ON")
endif()

# =========================================================
# --- MISC. BUILD OPTIONS ---
# =========================================================
option(AUTO_DETECT_DEVICES "Automatically detect available devices" ON)

if(AUTO_DETECT_DEVICES)
    message(STATUS "Auto-detecting available devices...")

    # NVIDIA
    file(GLOB NVIDIA_DEV_FILES "/dev/nvidia*")

    if(NVIDIA_DEV_FILES)
        set(WITH_NVIDIA ON)
        message(STATUS "Auto-detected NVIDIA environment.")
    endif()

    # MetaX
    if(DEFINED ENV{MACA_PATH})
        set(WITH_METAX ON)
        message(STATUS "Auto-detected MetaX environment from MACA_PATH")
    else()
        execute_process(
            COMMAND sh -c "grep -h 9999 /sys/bus/pci/devices/*/vendor 2>/dev/null"
            OUTPUT_VARIABLE _pci_vendor_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string(FIND "${_pci_vendor_output}" "9999" _found_pos)

        if(_found_pos GREATER -1)
            set(WITH_METAX ON)
            message(STATUS "Detected MetaX GPU from PCI vendor ID 0x9999")
        else()
            set(WITH_METAX OFF)
            message(STATUS "No MetaX GPU detected")
        endif()
    endif()
endif()

# =========================================================
# --- DEPENDENCY DISCOVERY ---
# =========================================================
if(WITH_NVIDIA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(WITH_METAX)
    set(MACA_PATH $ENV{MACA_PATH})
    if(NOT MACA_PATH)
        # Normally can be found at: `/opt/maca/`.
        set(MACA_PATH "/opt/maca")
        message(WARNING "MACA_PATH environment variable not set. Defaulting to /opt/maca/. Please verify this path is correct.")
    endif()

    set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mxcc_wrapper.sh)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/mxcc_wrapper.sh)

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
endif()

if(WITH_OMPI)
    find_package(MPI REQUIRED)
endif()

if(WITH_NCCL)
    if (NOT WITH_NVIDIA)
        message(FATAL_ERROR "NCCL backend requires NVIDIA GPU support. Please enable WITH_NVIDIA.")
    endif()

    find_library(NCCL_LIB NAMES nccl REQUIRED)
    find_path(NCCL_INC NAMES nccl.h REQUIRED)

    include_directories(${NCCL_INC})
endif()

# Python is required for code generation
find_package(Python3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(examples)
