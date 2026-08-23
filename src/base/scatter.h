#ifndef INFINI_CCL_BASE_SCATTER_H_
#define INFINI_CCL_BASE_SCATTER_H_

#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct ScatterImpl;

class Scatter : public Operation<Scatter> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(const void *send_buff, void *recv_buff,
                              size_t count, DataType datatype, int root,
                              void *comm_handle, void *stream) {
    if (HasInvalidRequiredArgs(datatype, root, comm_handle)) {
      return ReturnStatus::kInvalidArgument;
    }
    if (count == 0) {
      return ReturnStatus::kSuccess;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    if (!recv_buff) {
      LOG("Invalid receive buffer pointer for `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }
    if (comm->rank() == root && !send_buff) {
      LOG("Invalid root send buffer pointer for `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }

    return ScatterImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, root, comm, stream);
  }

 private:
  static bool HasInvalidRequiredArgs(DataType datatype, int root,
                                     void *comm_handle) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Scatter`.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Scatter`.");
      return true;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    if (root < 0 || root >= comm->size()) {
      LOG("Invalid root rank for `Scatter`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_SCATTER_H_
