#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_RANK_H_

#include "backends/ccl/common/impl/comm_init_rank.h"

namespace infini::ccl {

template <Device::Type device>
class CommInitRankImpl<BackendType::kCncl, device>
    : public CclCommInitRankImpl<BackendType::kCncl, device> {};

template <>
struct BackendEnabled<CommInitRank, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_RANK_H_
