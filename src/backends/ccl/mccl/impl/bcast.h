#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_BCAST_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_BCAST_H_

#include "base/bcast.h"

namespace infini::ccl {

template <>
struct BackendEnabled<Bcast, BackendType::kMccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_IMPL_BCAST_H_
