#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_GATHER_H_

#include "backends/ccl/common/impl/collectives.h"
#include "base/gather.h"

namespace infini::ccl {

template <Device::Type device>
struct GatherImpl<BackendType::kMccl, device>
    : CclGatherImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<Gather, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_GATHER_H_
