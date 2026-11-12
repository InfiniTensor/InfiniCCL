#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_GATHER_H_

#include "backends/ccl/common/impl/gather.h"

namespace infini::ccl {

template <Device::Type device>
class GatherImpl<BackendType::kMccl, device>
    : public CclGatherImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<Gather, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_GATHER_H_
