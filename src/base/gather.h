#ifndef INFINI_CCL_BASE_GATHER_H_
#define INFINI_CCL_BASE_GATHER_H_

#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct GatherImpl;

class Gather : public Operation<Gather> {
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
    if (!send_buff) {
      LOG("Invalid send buffer pointer for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    if (comm->rank() == root && !recv_buff) {
      LOG("Invalid root receive buffer pointer for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }

    return GatherImpl<backend_type, device_type>::Apply(
        send_buff, recv_buff, count, datatype, root, comm, stream);
  }

 private:
  static bool HasInvalidRequiredArgs(DataType datatype, int root,
                                     void *comm_handle) {
    if (!comm_handle) {
      LOG("Invalid communicator handle for `Gather`.");
      return true;
    }
    if (datatype < DataType::kChar || datatype >= DataType::kNumTypes) {
      LOG("Invalid data type for `Gather`.");
      return true;
    }
    auto *comm = static_cast<Communicator *>(comm_handle);
    if (root < 0 || root >= comm->size()) {
      LOG("Invalid root rank for `Gather`.");
      return true;
    }
    return false;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_GATHER_H_
