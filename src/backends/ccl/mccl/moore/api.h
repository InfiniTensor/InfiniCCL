#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_MOORE_API_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_MOORE_API_H_

#include "backends/ccl/mccl/api.h"
#include "devices/moore/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kMccl, Device::Type::kMoore>
    : McclApi<Device::Type::kMoore> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_MOORE_API_H_
