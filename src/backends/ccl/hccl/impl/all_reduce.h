#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_ALL_REDUCE_H_

#include "backends/ccl/common/impl/all_reduce.h"

namespace infini::ccl {

template <Device::Type device>
class AllReduceImpl<BackendType::kHccl, device>
    : public CclAllReduceImpl<BackendType::kHccl, device> {};

template <>
struct BackendEnabled<AllReduce, BackendType::kHccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_ALL_REDUCE_H_
