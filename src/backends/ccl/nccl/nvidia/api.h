#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_

#include "devices/nvidia/runtime_.h"
#include "backends/ccl/nccl/api.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kNccl, Device::Type::kNvidia>
    : NcclApi<Device::Type::kNvidia> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_
