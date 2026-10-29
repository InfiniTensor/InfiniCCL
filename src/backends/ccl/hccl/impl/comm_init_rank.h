#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_COMM_INIT_RANK_H_

#include "backends/ccl/common/impl/comm_init_rank.h"

namespace infini::ccl {

template <Device::Type device>
class CommInitRankImpl<BackendType::kHccl, device>
    : public CclCommInitRankImpl<BackendType::kHccl, device> {};

template <>
struct BackendEnabled<CommInitRank, BackendType::kHccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_IMPL_COMM_INIT_RANK_H_
