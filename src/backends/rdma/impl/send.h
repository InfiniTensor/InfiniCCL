#ifndef INFINI_CCL_BACKENDS_RDMA_IMPL_SEND_H_
#define INFINI_CCL_BACKENDS_RDMA_IMPL_SEND_H_

#include <limits>

#include "backends/rdma/comm_instance.h"
#include "base/send.h"

namespace infini::ccl {

template <Device::Type device_type>
class SendImpl<BackendType::kRdma, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, size_t count,
                            DataType data_type, int peer, Communicator *comm,
                            void *stream) {
    if (!comm || comm->inter_comm_backend() != BackendType::kRdma) {
      return ReturnStatus::kInvalidUsage;
    }
    const size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *instance = static_cast<RdmaInstance *>(comm->inter_comm());
    return instance->Transfer(const_cast<void *>(send_buff), count * type_size,
                              peer, false, device_type, stream);
  }
};

template <>
struct BackendEnabled<Send, BackendType::kRdma> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_IMPL_SEND_H_
