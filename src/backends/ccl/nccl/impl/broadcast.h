#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_BROADCAST_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_BROADCAST_H_

#include "backends/ccl/common/impl/broadcast.h"

namespace infini::ccl {

template <Device::Type device>
class BroadcastImpl<BackendType::kNccl, device>
    : public CclBroadcastImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<Broadcast, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_BROADCAST_H_
