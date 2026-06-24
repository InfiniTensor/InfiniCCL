#ifndef INFINI_CCL_NVIDIA_DEVICE__H_
#define INFINI_CCL_NVIDIA_DEVICE__H_

#include <cuda_runtime.h>

#include "cuda/checks.h"
#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kNvidia> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kNvidia>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  cudaPointerAttributes attr;
  INFINI_CHECK_CUDA(cudaPointerGetAttributes(&attr, ptr));

  if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
    return MemorySpace::kDevice;
  }

  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif
