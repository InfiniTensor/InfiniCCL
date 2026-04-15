#ifndef INFINI_CCL_OMPI_IMPL_COMM_DESTROY_H_
#define INFINI_CCL_OMPI_IMPL_COMM_DESTROY_H_

#include "base/comm_destroy.h"
#include "communicator.h"
#include "ompi/checks.h"
#include "ompi/comm_instance.h"

namespace infini::ccl {

template <Device::Type device_type>
class CommDestroyImpl<BackendType::kOmpi, device_type> {
public:
  static ReturnStatus Apply(void *comm) {
    auto *comm_internal = static_cast<Communicator *>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }

    if (auto *inter =
            static_cast<OmpiInstance *>(comm_internal->inter_comm())) {
      inter->Destroy();
    }
    comm_internal->set_inter_comm(nullptr);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommDestroy, BackendType::kOmpi> : std::true_type {};

} // namespace infini::ccl

#endif // INFINI_CCL_OMPI_IMPL_COMM_DESTROY_H_
