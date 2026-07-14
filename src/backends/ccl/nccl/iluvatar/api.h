#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_ILUVATAR_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_ILUVATAR_API_H_

#include "devices/iluvatar/runtime_.h"
#include "backends/ccl/nccl/api.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kNccl, Device::Type::kIluvatar>
    : NcclApi<Device::Type::kIluvatar> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_ILUVATAR_API_H_
