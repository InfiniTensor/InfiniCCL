#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_SCATTER_H_

#include "backends/ccl/common/impl/collectives.h"
#include "base/scatter.h"

namespace infini::ccl {

template <Device::Type device>
struct ScatterImpl<BackendType::kMccl, device>
    : CclScatterImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<Scatter, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_SCATTER_H_
