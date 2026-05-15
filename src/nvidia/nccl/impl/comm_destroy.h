#ifndef INFINI_CCL_NVIDIA_NCCL_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_NVIDIA_NCCL_IMPL_COMM_DESTROY_H_

#include "base/comm_destroy.h"
#include "communicator.h"
#include "nvidia/nccl/checks.h"
#include "nvidia/nccl/comm_instance.h"

namespace infini::ccl {

template <> class CommDestroyImpl<BackendType::kNccl, Device::Type::kNvidia> {
public:
  static ReturnStatus Apply(void *comm) {
    auto *comm_internal = static_cast<Communicator *>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }

    if (auto *intra =
            static_cast<NcclInstance *>(comm_internal->intra_comm())) {
      intra->Destroy();
    }

    comm_internal->set_intra_comm(nullptr);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommDestroy, BackendType::kNccl> : std::true_type {};

} // namespace infini::ccl

#endif // INFINI_CCL_NVIDIA_NCCL_IMPL_COMM_DESTROY_H_
