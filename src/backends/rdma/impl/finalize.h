#ifndef INFINI_CCL_BACKENDS_RDMA_IMPL_FINALIZE_H_
#define INFINI_CCL_BACKENDS_RDMA_IMPL_FINALIZE_H_

#include "base/finalize.h"

namespace infini::ccl {

template <Device::Type device_type>
class FinalizeImpl<BackendType::kRdma, device_type> {
 public:
  static ReturnStatus Apply() { return ReturnStatus::kSuccess; }
};

template <>
struct BackendEnabled<Finalize, BackendType::kRdma> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_IMPL_FINALIZE_H_
