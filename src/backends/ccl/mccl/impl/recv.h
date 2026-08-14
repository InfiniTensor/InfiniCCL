#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_RECV_H_

#include "backends/ccl/common/impl/recv.h"

namespace infini::ccl {

template <Device::Type device>
class RecvImpl<BackendType::kMccl, device>
    : public CclRecvImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<Recv, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_RECV_H_
