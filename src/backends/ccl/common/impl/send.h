#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SEND_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SEND_H_

#include "backends/ccl/common/impl/point_to_point.h"
#include "base/send.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclSendImpl {
 public:
  static ReturnStatus Apply(const void *send_buff, size_t count,
                            DataType data_type, int peer, Communicator *comm,
                            void *stream) {
    using Api = CclApi<backend, device>;
    using PointToPoint = CclPointToPoint<backend, device>;

    if (!PointToPoint::HasNativeCommunicator(comm)) {
      if (!comm || !comm->inter_comm() ||
          comm->inter_comm_backend() != BackendType::kOmpi) {
        return ReturnStatus::kInternalError;
      }

      if constexpr (BackendEnabled<Send, BackendType::kOmpi>::value) {
        return SendImpl<BackendType::kOmpi, device>::Apply(
            send_buff, count, data_type, peer, comm, stream);
      }

      return ReturnStatus::kInternalError;
    }

    return PointToPoint::Call(
        data_type, comm, [&](auto native_type, auto native_comm) {
          return Api::Send(send_buff, count, native_type, peer, native_comm,
                           reinterpret_cast<typename Api::Stream>(stream));
        });
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SEND_H_
