#ifndef INFINI_CCL_BACKENDS_IBV_IMPL_INIT_H_
#define INFINI_CCL_BACKENDS_IBV_IMPL_INIT_H_

#include "base/init.h"

namespace infini::ccl {

template <Device::Type device_type>
class InitImpl<BackendType::kIbv, device_type> {
 public:
  static ReturnStatus Apply(int* /*argc*/, char*** /*argv*/) {
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Init, BackendType::kIbv> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_IMPL_INIT_H_
