#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_ALL_TO_ALL_H_

#include "backends/ccl/common/impl/collectives.h"
#include "base/all_to_all.h"

namespace infini::ccl {

template <Device::Type device>
struct AllToAllImpl<BackendType::kMccl, device>
    : CclAllToAllImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<AllToAll, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_ALL_TO_ALL_H_
