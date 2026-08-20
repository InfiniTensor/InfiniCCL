#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_ALL_TO_ALL_H_

#include "backends/ccl/common/impl/all_to_all.h"

namespace infini::ccl {

template <Device::Type device>
class AllToAllImpl<BackendType::kMccl, device>
    : public CclAllToAllImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<AllToAll, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_ALL_TO_ALL_H_
