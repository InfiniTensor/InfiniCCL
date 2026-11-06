#include "devices/nvidia/rdma_memory.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace infini::ccl {
namespace {

ReturnStatus CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA operation `" << operation
              << "` failed: " << cudaGetErrorString(status) << ".\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus CheckDriver(CUresult status, const char *operation) {
  if (status != CUDA_SUCCESS) {
    const char *message = "Unknown CUDA driver error";
    (void)cuGetErrorString(status, &message);
    std::cerr << "CUDA operation `" << operation << "` failed: " << message
              << ".\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus ValidateDeviceBuffer(void *buffer, size_t bytes) {
  const auto pointer = reinterpret_cast<CUdeviceptr>(buffer);
  CUdeviceptr base = 0;
  size_t allocation_bytes = 0;
  auto status =
      CheckDriver(cuMemGetAddressRange(&base, &allocation_bytes, pointer),
                  "cuMemGetAddressRange");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  if (pointer < base || pointer - base > allocation_bytes ||
      bytes > allocation_bytes - (pointer - base)) {
    return ReturnStatus::kInvalidArgument;
  }
  unsigned int sync_memops = 1;
  return CheckDriver(
      cuPointerSetAttribute(&sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                            pointer),
      "cuPointerSetAttribute(SYNC_MEMOPS)");
}

ReturnStatus QueryFlushPolicy(int device, bool *flush) {
  int supported = 0;
  int ordering = 0;
  int options = 0;
  auto status =
      CheckCuda(cudaDeviceGetAttribute(
                    &supported, cudaDevAttrGPUDirectRDMASupported, device),
                "cudaDeviceGetAttribute(GPUDirectRDMASupported)");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  if (!supported) {
    std::cerr << "The CUDA device does not report GPUDirect RDMA support.\n";
    return ReturnStatus::kNotSupported;
  }
  status =
      CheckCuda(cudaDeviceGetAttribute(
                    &ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device),
                "cudaDeviceGetAttribute(GPUDirectRDMAWritesOrdering)");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  *flush = ordering < cudaFlushGPUDirectRDMAWritesToOwner;
  if (!*flush) {
    return ReturnStatus::kSuccess;
  }
  status = CheckCuda(
      cudaDeviceGetAttribute(
          &options, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device),
      "cudaDeviceGetAttribute(GPUDirectRDMAFlushWritesOptions)");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  if (!(options & cudaFlushGPUDirectRDMAWritesOptionHost)) {
    std::cerr << "NVIDIA RDMA requires native owner ordering or Host flush "
                 "support.\n";
    return ReturnStatus::kNotSupported;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus RegisterDeviceMemory(ibv_pd *pd, void *buffer, size_t bytes,
                                  RdmaMemoryRegion *region) {
  if (!pd || !buffer || bytes == 0 || !region || region->mr ||
      region->fd >= 0) {
    return ReturnStatus::kInvalidArgument;
  }
  region->mr = ibv_reg_mr(pd, buffer, bytes, IBV_ACCESS_LOCAL_WRITE);
  if (!region->mr) {
    const int error = errno;
    std::cerr << "NVIDIA peer-memory MR registration failed: "
              << std::strerror(error) << " (" << error
              << "). Check `nvidia-peermem` and GPU/NIC topology; no Host "
                 "staging is used.\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

class NvidiaMemoryAdapter : public RdmaMemoryAdapter {
 public:
  ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) override;

  ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                        RdmaMemoryRegion *region) override {
    return device_memory_ ? RegisterDeviceMemory(pd, buffer, bytes, region)
                          : RegisterRdmaHostMemory(pd, buffer, bytes, region);
  }

  ReturnStatus Complete(bool receive) override {
    if (receive && device_memory_ && flush_) {
      return CheckCuda(cudaDeviceFlushGPUDirectRDMAWrites(
                           cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                           cudaFlushGPUDirectRDMAWritesToOwner),
                       "cudaDeviceFlushGPUDirectRDMAWrites");
    }
    return ReturnStatus::kSuccess;
  }

 private:
  bool device_memory_ = false;

  bool flush_ = false;
};

ReturnStatus NvidiaMemoryAdapter::Prepare(void *buffer, size_t bytes,
                                          void *stream) {
  cudaPointerAttributes attributes{};
  auto status = CheckCuda(cudaPointerGetAttributes(&attributes, buffer),
                          "cudaPointerGetAttributes");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  device_memory_ = attributes.type == cudaMemoryTypeDevice;
  if (!device_memory_ && attributes.type != cudaMemoryTypeHost &&
      attributes.type != cudaMemoryTypeUnregistered) {
    std::cerr << "Managed or array memory is not supported by NVIDIA RDMA.\n";
    return ReturnStatus::kNotSupported;
  }
  if (device_memory_) {
    int device = -1;
    status = CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    if (attributes.device != device) {
      std::cerr
          << "The RDMA buffer does not belong to the current CUDA device.\n";
      return ReturnStatus::kInvalidArgument;
    }
    status = ValidateDeviceBuffer(buffer, bytes);
    if (status == ReturnStatus::kSuccess) {
      status = QueryFlushPolicy(device, &flush_);
    }
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
  }
  return CheckCuda(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
      "cudaStreamSynchronize");
}

}  // namespace

ReturnStatus MemoryAdapterFactory<Device::Type::kNvidia>::Create(
    std::unique_ptr<RdmaMemoryAdapter> *adapter) {
  if (!adapter) {
    return ReturnStatus::kInvalidArgument;
  }
  *adapter = std::make_unique<NvidiaMemoryAdapter>();
  return ReturnStatus::kSuccess;
}

}  // namespace infini::ccl
