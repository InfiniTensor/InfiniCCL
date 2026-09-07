#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_ASCEND_API_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_ASCEND_API_H_

#include "backends/ccl/hccl/api.h"
#include "devices/ascend/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kHccl, Device::Type::kAscend>
    : HcclApi<Device::Type::kAscend> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_ASCEND_API_H_
