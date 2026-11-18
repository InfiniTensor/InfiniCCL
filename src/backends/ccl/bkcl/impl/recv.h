#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_RECV_H_

#include "backends/ccl/common/impl/recv.h"

namespace infini::ccl {

template <Device::Type device>
class RecvImpl<BackendType::kBkcl, device>
    : public CclRecvImpl<BackendType::kBkcl, device> {};

template <>
struct BackendEnabled<Recv, BackendType::kBkcl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_RECV_H_
