#ifndef INFINI_CCL_BACKENDS_IBV_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_IBV_IMPL_RECV_H_

#include <limits>

#include "backends/ibv/comm_instance.h"
#include "base/recv.h"

namespace infini::ccl {

template <Device::Type device_type>
class RecvImpl<BackendType::kIbv, device_type> {
 public:
  static ReturnStatus Apply(void *recv_buff, size_t count, DataType data_type,
                            int peer, Communicator *comm, void *stream) {
    if (!comm || comm->inter_comm_backend() != BackendType::kIbv) {
      return ReturnStatus::kInvalidUsage;
    }
    const size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *instance = static_cast<IbvInstance *>(comm->inter_comm());
    return instance->Transfer(recv_buff, count * type_size, peer, true,
                              device_type, stream);
  }
};

template <>
struct BackendEnabled<Recv, BackendType::kIbv> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_IMPL_RECV_H_
