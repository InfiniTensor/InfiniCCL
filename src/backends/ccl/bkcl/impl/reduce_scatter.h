#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_REDUCE_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_REDUCE_SCATTER_H_

#include "backends/ccl/common/impl/reduce_scatter.h"

namespace infini::ccl {

template <Device::Type device>
class ReduceScatterImpl<BackendType::kBkcl, device>
    : public CclReduceScatterImpl<BackendType::kBkcl, device> {};

template <>
struct BackendEnabled<ReduceScatter, BackendType::kBkcl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_REDUCE_SCATTER_H_
