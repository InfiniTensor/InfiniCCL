cmake_minimum_required(VERSION 3.18)
project(InfiniCCL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# --- DEVICE OPTIONS (Hardware Runtimes) ---
# =========================================================
option(WITH_NVIDIA      "Enable NVIDIA GPU support"     OFF)
option(WITH_METAX       "Enable MetaX GPU support"      OFF)
option(WITH_CAMBRICON   "Enable Cambricon MLU support"  OFF)

set(WITH_CPU ON CACHE INTERNAL "CPU backend is always enabled")

# =========================================================
# --- BACKEND OPTIONS (Communication Protocols) ---
# =========================================================
option(WITH_OMPI "Enable OpenMPI backend" OFF)
option(WITH_MPICH "Enable MPICH backend" OFF)
option(WITH_NCCL "Enable NCCL backend"    OFF)

# =========================================================
# --- MISC. BUILD OPTIONS ---
# =========================================================
option(AUTO_DETECT_DEVICES "Automatically detect available devices" ON)
option(AUTO_DETECT_BACKENDS "Automatically detect available backends" OFF)
option(BUILD_EXAMPLES "Build internal examples" ON)

# =========================================================
# --- AUTO-DETECTION: DEVICES ---
# =========================================================
if(AUTO_DETECT_DEVICES)
    message(STATUS "Auto-detecting available devices...")

    # NVIDIA
    set(NVIDIA_FOUND FALSE)

    file(GLOB NVIDIA_DEV_FILES "/dev/nvidia*")

    if(NVIDIA_DEV_FILES)
        set(NVIDIA_FOUND TRUE)
    elseif(NOT NVIDIA_FOUND)
        find_program(NVIDIA_SMI_PATH nvidia-smi)
        if(NVIDIA_SMI_PATH)
            execute_process(
                COMMAND ${NVIDIA_SMI_PATH} -L
                RESULT_VARIABLE SMI_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(SMI_RESULT EQUAL 0)
                set(NVIDIA_FOUND TRUE)
            endif()
        endif()
    endif()

    if(NVIDIA_FOUND)
        set(WITH_NVIDIA ON)
        message(STATUS "Auto-detected NVIDIA environment.")
    else()
        message(STATUS "NVIDIA environment not detected.")
    endif()

    # MetaX
    if(DEFINED ENV{MACA_PATH})
        set(WITH_METAX ON)
        message(STATUS "Auto-detected MetaX environment from `MACA_PATH`")
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

    # Cambricon
    set(CAMBRICON_FOUND FALSE)

    if(DEFINED ENV{NEUWARE_HOME})
        set(CAMBRICON_FOUND TRUE)
        message(STATUS "Auto-detected Cambricon environment from `NEUWARE_HOME`")
    else()
        find_program(CNMON_PATH cnmon)
        if(CNMON_PATH)
            execute_process(
                COMMAND ${CNMON_PATH} -l
                RESULT_VARIABLE CNMON_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(CNMON_RESULT EQUAL 0)
                set(CAMBRICON_FOUND TRUE)
            endif()
        endif()
    endif()

    if(CAMBRICON_FOUND)
        set(WITH_CAMBRICON ON)
        message(STATUS "Cambricon environment detected.")
    else()
        set(WITH_CAMBRICON OFF)
        message(STATUS "Cambricon environment not detected.")
    endif()
endif()

# =========================================================
# --- AUTO-DETECTION: BACKENDS ---
# =========================================================
if(AUTO_DETECT_BACKENDS)
    message(STATUS "Auto-detecting available communication backends...")

    # Detect MPI Dependencies
    find_program(MPIRUN_PATH NAMES mpirun mpiexec)
    find_program(MPICC_PATH NAMES mpicc)
    
    if(MPIRUN_PATH AND MPICC_PATH)
        # Query the active `mpirun` variant.
        execute_process(
            COMMAND ${MPIRUN_PATH} --version
            OUTPUT_VARIABLE MPI_VERSION_OUT
            ERROR_VARIABLE MPI_VERSION_ERR
        )
        set(MPI_VERSION_INFO "${MPI_VERSION_OUT}${MPI_VERSION_ERR}")

        if(MPI_VERSION_INFO MATCHES "Open MPI" OR MPI_VERSION_INFO MATCHES "PRRTE")
            set(WITH_OMPI ON)
            message(STATUS "Auto-detected OpenMPI environment.")
        elseif(MPI_VERSION_INFO MATCHES "MPICH" OR MPI_VERSION_INFO MATCHES "Hydra" OR MPI_VERSION_INFO MATCHES "Intel\\(R\\) MPI")
            set(WITH_MPICH ON)
            message(STATUS "Auto-detected MPICH/Hydra environment.")
        else()
            # Fallback to OMPI if it's an unrecognized wrapper.
            set(WITH_OMPI ON)
            message(STATUS "Unknown MPI implementation. Defaulting to `WITH_OMPI=ON`")
        endif()
    endif()

    # Detect NCCL Dependencies
    if(WITH_NVIDIA)
        find_path(AUTO_NCCL_INC NAMES nccl.h QUIET)
        find_library(AUTO_NCCL_LIB NAMES nccl QUIET)

        if(AUTO_NCCL_INC AND AUTO_NCCL_LIB)
            set(WITH_NCCL ON)
            message(STATUS "Auto-detected NCCL backend.")
        else()
            message(STATUS "NCCL library/headers not found in standard paths.")
        endif()
    else()
        message(STATUS "No suitable device environment, skipping NCCL detection.")
    endif()
endif()

# Fallback: If no backends are enabled or auto-detected, fall back to OpenMPI as the default bootstrap profile.
if(NOT WITH_OMPI AND NOT WITH_MPICH AND NOT WITH_NCCL)
    set(WITH_OMPI ON)
    message(STATUS "No backend specified or detected. Defaulting to `WITH_OMPI=ON`")
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

    get_filename_component(MXCC_WRAPPER_ABS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/devices/mxcc_wrapper.sh" ABSOLUTE)
    
    set(CMAKE_C_COMPILER   "${MXCC_WRAPPER_ABS}")
    set(CMAKE_CXX_COMPILER "${MXCC_WRAPPER_ABS}")

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
endif()

if(WITH_CAMBRICON)
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})

    include_directories("${NEUWARE_HOME}/include")
    link_directories("${NEUWARE_HOME}/lib")
    link_directories("${NEUWARE_HOME}/lib64")

    # Libraries: `cnrt` / `cnnl` / `cnnl_extra` / `cnpapi`.
    find_library(CAMBRICON_RUNTIME_LIB    NAMES cnrt       HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_LIB       NAMES cnnl       HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_CNNL_EXTRA_LIB NAMES cnnl_extra HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
    find_library(CAMBRICON_PAPI_LIB       NAMES cnpapi     HINTS "${NEUWARE_HOME}/lib64" REQUIRED)
endif()

if(WITH_OMPI OR WITH_MPICH)
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

install(TARGETS infiniccl DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/infiniccl)

install(PROGRAMS scripts/icclrun.py DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME icclrun)

# Install the supporting logic to a private folder.
set(ICCL_PRIVATE_LIB ${CMAKE_INSTALL_LIBDIR}/infiniccl)
install(FILES scripts/icclrun_logic.py DESTINATION ${ICCL_PRIVATE_LIB})
install(DIRECTORY scripts/backends DESTINATION ${ICCL_PRIVATE_LIB})
