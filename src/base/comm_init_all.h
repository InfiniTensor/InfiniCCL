#ifndef INFINI_CCL_BASE_COMM_INIT_ALL_H_
#define INFINI_CCL_BASE_COMM_INIT_ALL_H_

#include "communicator.h"
#include "logging.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct CommInitAllImpl;

class CommInitAll : public Operation<CommInitAll> {
 public:
  template <BackendType backend_type, Device::Type device_type>
  static ReturnStatus Execute(void** comm_handles, int n_dev,
                              const int* dev_list) {
    if (!comm_handles || n_dev <= 0) {
      LOG("Invalid communicator handle for `CommInitAll`.");
      return ReturnStatus::kInvalidArgument;
    }

    return CommInitAllImpl<backend_type, device_type>::Apply(comm_handles,
                                                             n_dev, dev_list);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_COMM_INIT_ALL_H_
