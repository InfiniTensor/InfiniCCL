#ifndef INFINI_CCL_MOORE_DATA_TYPE__H_
#define INFINI_CCL_MOORE_DATA_TYPE__H_

// clang-format off
#include <musa_bf16.h>
#include <musa_fp16.h>
// clang-format on

#include "data_type_impl.h"
#include "moore/device_.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kMoore, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kMoore, DataType::kBFloat16> {
  using type = __mt_bfloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_MOORE_DATA_TYPE__H_
