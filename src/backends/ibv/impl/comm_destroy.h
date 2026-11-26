#ifndef INFINI_CCL_BACKENDS_IBV_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_BACKENDS_IBV_IMPL_COMM_DESTROY_H_

#include "backends/ibv/comm_instance.h"
#include "base/comm_destroy.h"
#include "communicator.h"

namespace infini::ccl {

template <Device::Type device_type>
class CommDestroyImpl<BackendType::kIbv, device_type> {
 public:
  static ReturnStatus Apply(void *comm) {
    auto *comm_internal = static_cast<Communicator *>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }
    if (!comm_internal->inter_comm()) {
      return ReturnStatus::kSuccess;
    }
    if (comm_internal->inter_comm_backend() != BackendType::kIbv) {
      return ReturnStatus::kInvalidUsage;
    }

    if (auto *inter =
            static_cast<IbvInstance *>(comm_internal->inter_comm())) {
      inter->Destroy();
    }
    comm_internal->set_inter_comm(nullptr);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommDestroy, BackendType::kIbv> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_IMPL_COMM_DESTROY_H_
