#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_POINT_TO_POINT_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_POINT_TO_POINT_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclPointToPoint {
 public:
  static bool HasNativeCommunicator(const Communicator *comm) {
    return comm && comm->intra_comm() &&
           comm->intra_comm_backend() == backend &&
           comm->device_type() == device;
  }

  template <typename Callback>
  static ReturnStatus Call(DataType data_type, Communicator *comm,
                           Callback callback) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    if (!HasNativeCommunicator(comm)) {
      return ReturnStatus::kInternalError;
    }

    auto *instance = static_cast<CommInstance *>(comm->intra_comm());
    if (!instance->handle) {
      return ReturnStatus::kInternalError;
    }

    typename Api::DataType native_type{};
    if (!TypeMap::ToBackendDataType(data_type, &native_type)) {
      return ReturnStatus::kNotSupported;
    }

    return Api::Check(callback(native_type, instance->handle));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_POINT_TO_POINT_H_
