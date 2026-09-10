#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_REDUCE_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_REDUCE_SCATTER_H_

#include "backends/ccl/common/impl/collectives.h"
#include "base/reduce_scatter.h"

namespace infini::ccl {

template <Device::Type device>
struct ReduceScatterImpl<BackendType::kMccl, device>
    : CclReduceScatterImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<ReduceScatter, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_REDUCE_SCATTER_H_
