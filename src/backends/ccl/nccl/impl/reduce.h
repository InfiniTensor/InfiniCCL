#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_REDUCE_H_

#include "backends/ccl/common/impl/reduce.h"

namespace infini::ccl {

template <Device::Type device>
class ReduceImpl<BackendType::kNccl, device>
    : public CclReduceImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<Reduce, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_REDUCE_H_
