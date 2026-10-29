#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_REDUCE_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_REDUCE_SCATTER_H_

#include "backends/ccl/common/impl/reduce_scatter.h"

namespace infini::ccl {

template <Device::Type device>
class ReduceScatterImpl<BackendType::kHccl, device>
    : public CclReduceScatterImpl<BackendType::kHccl, device> {};

template <>
struct BackendEnabled<ReduceScatter, BackendType::kHccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_REDUCE_SCATTER_H_
