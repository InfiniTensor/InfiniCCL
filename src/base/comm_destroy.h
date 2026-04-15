#ifndef INFINI_CCL_BASE_COMM_DESTROY_H_
#define INFINI_CCL_BASE_COMM_DESTROY_H_

#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend_type, Device::Type device_type>
struct CommDestroyImpl;

class CommDestroy : public Operation<CommDestroy> {
public:
  template <BackendType backend_type, Device::Type device_type,
            typename... Args>
  static ReturnStatus Execute(Args &&...args) {
    return CommDestroyImpl<backend_type, device_type>::Apply(
        std::forward<Args>(args)...);
  }
};

} // namespace infini::ccl

#endif // INFINI_CCL_BASE_COMM_DESTROY_H_
