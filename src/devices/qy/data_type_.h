#ifndef INFINI_CCL_DEVICES_QY_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_QY_DATA_TYPE_H_

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// clang-format on

#include "data_type_impl.h"
#include "devices/qy/device_.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kQy, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kQy, DataType::kBFloat16> {
  using type = __nv_bfloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_QY_DATA_TYPE_H_
