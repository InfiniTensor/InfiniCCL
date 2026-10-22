include_guard(GLOBAL)

function(infiniccl_detect_iluvatar output)
    set(_found FALSE)
    file(GLOB _device_files "/dev/iluvatar*")

    if(_device_files)
        set(_found TRUE)
    else()
        find_program(_INFINICCL_ILUVATAR_SMI_PATH ixsmi)
        if(_INFINICCL_ILUVATAR_SMI_PATH)
            execute_process(
                COMMAND ${_INFINICCL_ILUVATAR_SMI_PATH} -L
                RESULT_VARIABLE _smi_result
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_smi_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
    endif()

    set(${output} ${_found} PARENT_SCOPE)
endfunction()

function(infiniccl_configure_iluvatar)
    find_program(_INFINICCL_ILUVATAR_CUDA_COMPILER NAMES clang++ HINTS /usr/local/corex/bin)
    if(NOT _INFINICCL_ILUVATAR_CUDA_COMPILER)
        message(FATAL_ERROR "`WITH_ILUVATAR` is `ON` but CoreX `clang++` was not found.")
    endif()

    get_filename_component(_cuda_bin_dir "${_INFINICCL_ILUVATAR_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_cuda_root "${_cuda_bin_dir}/.." ABSOLUTE)
    set(CUDAToolkit_ROOT "${_cuda_root}" CACHE PATH "Iluvatar CoreX toolkit root" FORCE)
    find_package(CUDAToolkit REQUIRED)

    set(_cuda_flags "-fPIC"
        "-Wno-error=unused-variable"
        "-Wno-error=unused-private-field"
        "-Wno-unused-variable"
        "-std=c++17"
        "--cuda-path=${_cuda_root}"
        "-x"
        "ivcore")
    set(ILUVATAR_CUDA_FLAGS "${_cuda_flags}"
        CACHE STRING "Iluvatar CUDA compiler flags")

    set(ILUVATAR_CUDA_COMPILER "${_INFINICCL_ILUVATAR_CUDA_COMPILER}" PARENT_SCOPE)
    set(ILUVATAR_CUDA_ROOT "${_cuda_root}" PARENT_SCOPE)
    set(ILUVATAR_INCLUDE_DIR "${_cuda_root}/include" PARENT_SCOPE)
    set(ILUVATAR_LIB_DIR "${_cuda_root}/lib;${_cuda_root}/lib64" PARENT_SCOPE)
    set(ILUVATAR_CUDA_FLAGS "${ILUVATAR_CUDA_FLAGS}" PARENT_SCOPE)
endfunction()
