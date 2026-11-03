#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_TO_ALL_H_

#include <nccl.h>

#include <cstddef>
#include <limits>

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/all_to_all.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device>
class AllToAllImpl<BackendType::kNccl, device> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            Communicator *comm, void *stream) {
    using Api = CclApi<BackendType::kNccl, device>;
    using TypeMap = CclTypeMap<BackendType::kNccl, device>;
    using CommInstance = CclCommInstance<Api>;

    if (!comm || !comm->intra_comm() ||
        comm->intra_comm_backend() != BackendType::kNccl ||
        comm->device_type() != device) {
      return ReturnStatus::kInternalError;
    }

    if (send_buff == recv_buff) {
      LOG("In-place buffers are not supported by the native CCL `AllToAll` "
          "implementation.");
      return ReturnStatus::kNotSupported;
    }
    if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size()) {
      LOG("Invalid rank or world size for native CCL `AllToAll`.");
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

    const size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Per-peer byte size overflows `size_t` for native CCL "
          "`AllToAll`.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t peer_bytes = count * type_size;
    const size_t world_size = static_cast<size_t>(comm->size());
    if (peer_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Total byte size overflows `size_t` for native CCL `AllToAll`.");
      return ReturnStatus::kInvalidArgument;
    }

    const auto *send_bytes = static_cast<const std::byte *>(send_buff);
    auto *recv_bytes = static_cast<std::byte *>(recv_buff);
    auto native_stream = reinterpret_cast<typename Api::Stream>(stream);

    ReturnStatus status = Api::Check(ncclGroupStart());
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    ReturnStatus first_error = ReturnStatus::kSuccess;
    for (int peer = 0; peer < comm->size(); ++peer) {
      const size_t offset = static_cast<size_t>(peer) * peer_bytes;
      status = Api::Check(ncclSend(send_bytes + offset, count, native_type,
                                   peer, instance->handle, native_stream));
      if (first_error == ReturnStatus::kSuccess &&
          status != ReturnStatus::kSuccess) {
        first_error = status;
      }

      status = Api::Check(ncclRecv(recv_bytes + offset, count, native_type,
                                   peer, instance->handle, native_stream));
      if (first_error == ReturnStatus::kSuccess &&
          status != ReturnStatus::kSuccess) {
        first_error = status;
      }
    }

    const ReturnStatus group_end_status = Api::Check(ncclGroupEnd());
    return first_error != ReturnStatus::kSuccess ? first_error
                                                 : group_end_status;
  }
};

template <>
struct BackendEnabled<AllToAll, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_TO_ALL_H_
