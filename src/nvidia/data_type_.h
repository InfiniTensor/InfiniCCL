#ifndef INFINI_CCL_NVIDIA_DATA_TYPE__H_
#define INFINI_CCL_NVIDIA_DATA_TYPE__H_

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// clang-format on

#include "data_type_impl.h"
#include "nvidia/device_.h"

namespace infini::ccl {

template <> struct TypeMap<Device::Type::kNvidia, DataType::kFloat16> {
  using type = half;
};

template <> struct TypeMap<Device::Type::kNvidia, DataType::kBFloat16> {
  using type = __nv_bfloat16;
};

} // namespace infini::ccl

#endif // INFINI_CCL_NVIDIA_DATA_TYPE__H_
