#ifndef INFINI_CCL_DEVICES_KUNLUN_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_KUNLUN_DATA_TYPE_H_

#include <cstring>

#include "data_type_impl.h"
#include "devices/kunlun/device_.h"

namespace infini::ccl {

// The kunlun 16-bit payload layout is identical to CPU's; reuse those
// definitions (the generated manifest includes both data_type_ headers in
// one TU, so redefining them here would break the build).
#include "devices/cpu/data_type_.h"

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kFloat16> {
  using type = Float16;
};

template <>
struct TypeMap<Device::Type::kKunlun, DataType::kBFloat16> {
  using type = BFloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_KUNLUN_DATA_TYPE_H_
