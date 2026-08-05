#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_BROADCAST_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_BROADCAST_H_

#include "backends/ccl/common/impl/broadcast.h"

namespace infini::ccl {

template <Device::Type device>
class BroadcastImpl<BackendType::kMccl, device>
    : public CclBroadcastImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<Broadcast, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_BROADCAST_H_
