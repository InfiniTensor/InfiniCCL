#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_SCATTER_H_

#include "backends/ccl/common/impl/scatter.h"

namespace infini::ccl {

template <Device::Type device>
class ScatterImpl<BackendType::kNccl, device>
    : public CclScatterImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<Scatter, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_SCATTER_H_
