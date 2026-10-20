#include "devices/nvidia/ibv_memory.h"

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

ReturnStatus ValidateDeviceBuffer(void *buffer, size_t bytes,
                                  IbvMemoryKey *key) {
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
  uint64_t buffer_id = 0;
  // A failed identity query only weakens cache validation to range matching.
  if (cuPointerGetAttribute(&buffer_id, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                            pointer) != CUDA_SUCCESS) {
    buffer_id = 0;
  }
  key->base = static_cast<uintptr_t>(base);
  key->bytes = allocation_bytes;
  key->tag = buffer_id;
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
    std::cerr << "The NVIDIA IBV backend requires native owner ordering or "
                 "Host flush support.\n";
    return ReturnStatus::kNotSupported;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus RegisterDeviceMemory(ibv_pd *pd, const IbvMemoryKey &key,
                                  IbvMemoryRegion *region) {
  if (!pd || !region || key.base == 0 || key.bytes == 0 || region->mr ||
      region->fd >= 0) {
    return ReturnStatus::kInvalidArgument;
  }
  // The whole allocation is registered once so subrange transfers reuse the
  // same memory region.
  region->mr = ibv_reg_mr(pd, reinterpret_cast<void *>(key.base), key.bytes,
                          IBV_ACCESS_LOCAL_WRITE);
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

class NvidiaMemoryAdapter : public IbvMemoryAdapter {
 public:
  ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) override;

  ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                        IbvMemoryRegion *region) override {
    return device_memory_ ? RegisterDeviceMemory(pd, key_, region)
                          : RegisterIbvHostMemory(pd, buffer, bytes, region);
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

  IbvMemoryKey memory_key() const override { return key_; }

 private:
  bool device_memory_ = false;

  bool flush_ = false;

  IbvMemoryKey key_;
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
    std::cerr << "Managed or array memory is not supported by the NVIDIA IBV "
                 "backend.\n";
    return ReturnStatus::kNotSupported;
  }
  key_ = IbvMemoryKey{reinterpret_cast<uintptr_t>(buffer), bytes, 0};
  if (device_memory_) {
    int device = -1;
    status = CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    if (attributes.device != device) {
      std::cerr
          << "The IBV buffer does not belong to the current CUDA device.\n";
      return ReturnStatus::kInvalidArgument;
    }
    status = ValidateDeviceBuffer(buffer, bytes, &key_);
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
    std::unique_ptr<IbvMemoryAdapter> *adapter) {
  if (!adapter) {
    return ReturnStatus::kInvalidArgument;
  }
  *adapter = std::make_unique<NvidiaMemoryAdapter>();
  return ReturnStatus::kSuccess;
}

}  // namespace infini::ccl
