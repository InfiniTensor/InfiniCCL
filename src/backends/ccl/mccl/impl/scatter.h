#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_SCATTER_H_

#include "backends/ccl/common/impl/scatter.h"

namespace infini::ccl {

template <Device::Type device>
class ScatterImpl<BackendType::kMccl, device>
    : public CclScatterImpl<BackendType::kMccl, device> {};

template <>
struct BackendEnabled<Scatter, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_SCATTER_H_
