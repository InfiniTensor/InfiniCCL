#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_TO_ALL_H_

#include "backends/ccl/common/impl/all_to_all.h"

namespace infini::ccl {

template <Device::Type device>
class AllToAllImpl<BackendType::kNccl, device>
    : public CclAllToAllImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<AllToAll, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_TO_ALL_H_
