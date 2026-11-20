#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_ALL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_ALL_GATHER_H_

#include "backends/ccl/common/impl/all_gather.h"

namespace infini::ccl {

template <Device::Type device>
class AllGatherImpl<BackendType::kBkcl, device>
    : public CclAllGatherImpl<BackendType::kBkcl, device> {};

template <>
struct BackendEnabled<AllGather, BackendType::kBkcl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_ALL_GATHER_H_
