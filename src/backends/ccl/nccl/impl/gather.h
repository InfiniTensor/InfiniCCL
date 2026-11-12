#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_GATHER_H_

#include "backends/ccl/common/impl/gather.h"

namespace infini::ccl {

template <Device::Type device>
class GatherImpl<BackendType::kNccl, device>
    : public CclGatherImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<Gather, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_GATHER_H_
