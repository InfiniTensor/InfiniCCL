#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_COMM_DESTROY_H_

#include "backends/ccl/common/impl/comm_destroy.h"

namespace infini::ccl {

template <Device::Type device>
class CommDestroyImpl<BackendType::kHccl, device>
    : public CclCommDestroyImpl<BackendType::kHccl, device> {};

template <>
struct BackendEnabled<CommDestroy, BackendType::kHccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_COMM_DESTROY_H_
