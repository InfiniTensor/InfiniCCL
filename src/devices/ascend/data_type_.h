#ifndef INFINI_CCL_DEVICES_ASCEND_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_ASCEND_DATA_TYPE_H_

#include <opdev/bfloat16.h>
#include <opdev/fp16_t.h>

#include "data_type_impl.h"
#include "devices/ascend/device_.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kAscend, DataType::kFloat16> {
  using type = op::fp16_t;
};

template <>
struct TypeMap<Device::Type::kAscend, DataType::kBFloat16> {
  using type = op::bfloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ASCEND_DATA_TYPE_H_
