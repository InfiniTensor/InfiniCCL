#ifndef INFINI_CCL_BACKENDS_RDMA_IMPL_INIT_H_
#define INFINI_CCL_BACKENDS_RDMA_IMPL_INIT_H_

#include "base/init.h"

namespace infini::ccl {

template <Device::Type device_type>
class InitImpl<BackendType::kRdma, device_type> {
 public:
  static ReturnStatus Apply(int* /*argc*/, char*** /*argv*/) {
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Init, BackendType::kRdma> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_IMPL_INIT_H_
