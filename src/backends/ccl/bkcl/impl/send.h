#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_SEND_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_SEND_H_

#include "backends/ccl/common/impl/send.h"

namespace infini::ccl {

template <Device::Type device>
class SendImpl<BackendType::kBkcl, device>
    : public CclSendImpl<BackendType::kBkcl, device> {};

template <>
struct BackendEnabled<Send, BackendType::kBkcl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_SEND_H_
