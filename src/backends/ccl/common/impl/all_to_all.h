#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_TO_ALL_H_

#include <cstddef>
#include <limits>

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/all_to_all.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclAllToAllImpl {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            Communicator *comm, void *stream) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    if (!comm || !comm->intra_comm() || comm->intra_comm_backend() != backend ||
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

    auto native_stream = reinterpret_cast<typename Api::Stream>(stream);
    ReturnStatus status = Api::Check(Api::GroupStart());
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    // Each peer receives its send block and supplies its receive block.
    ReturnStatus first_error = ReturnStatus::kSuccess;
    const auto *send_bytes = static_cast<const char *>(send_buff);
    auto *recv_bytes = static_cast<char *>(recv_buff);
    for (int peer = 0; peer < comm->size(); ++peer) {
      const size_t offset = static_cast<size_t>(peer) * peer_bytes;
      status = Api::Check(Api::Send(send_bytes + offset, count, native_type,
                                    peer, instance->handle, native_stream));
      if (status != ReturnStatus::kSuccess &&
          first_error == ReturnStatus::kSuccess) {
        first_error = status;
      }
      status = Api::Check(Api::Recv(recv_bytes + offset, count, native_type,
                                    peer, instance->handle, native_stream));
      if (status != ReturnStatus::kSuccess &&
          first_error == ReturnStatus::kSuccess) {
        first_error = status;
      }
    }

    // Always close a successfully started group, even after a send/recv error.
    const ReturnStatus group_end_status = Api::Check(Api::GroupEnd());
    return first_error != ReturnStatus::kSuccess ? first_error
                                                 : group_end_status;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_TO_ALL_H_
