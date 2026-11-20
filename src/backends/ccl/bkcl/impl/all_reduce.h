#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_ALL_REDUCE_H_

#include "backends/ccl/common/impl/all_reduce.h"

namespace infini::ccl {

template <Device::Type device>
class AllReduceImpl<BackendType::kBkcl, device>
    : public CclAllReduceImpl<BackendType::kBkcl, device> {};

template <>
struct BackendEnabled<AllReduce, BackendType::kBkcl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_ALL_REDUCE_H_
