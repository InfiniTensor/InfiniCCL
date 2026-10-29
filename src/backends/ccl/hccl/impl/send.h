#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_SEND_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_SEND_H_

#include "backends/ccl/common/impl/send.h"

namespace infini::ccl {

template <Device::Type device>
class SendImpl<BackendType::kHccl, device>
    : public CclSendImpl<BackendType::kHccl, device> {};

template <>
struct BackendEnabled<Send, BackendType::kHccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_SEND_H_
