#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_GATHER_H_

#include <cstddef>
#include <limits>

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/gather.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <BackendType>
struct DeferredGather {
  using type = Gather;
};

template <BackendType backend, Device::Type device>
class CclGatherImpl {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    const bool has_native_comm = comm && comm->intra_comm() &&
                                 comm->intra_comm_backend() == backend &&
                                 comm->device_type() == device;
    if (!has_native_comm) {
      if (!comm || !comm->inter_comm() ||
          comm->inter_comm_backend() != BackendType::kOmpi) {
        return ReturnStatus::kInternalError;
      }

      using FallbackOperation = typename DeferredGather<backend>::type;
      if constexpr (BackendEnabled<FallbackOperation,
                                   BackendType::kOmpi>::value) {
        return GatherImpl<BackendType::kOmpi, device>::Apply(
            send_buff, recv_buff, count, data_type, root, comm, stream);
      }

      return ReturnStatus::kInternalError;
    }

    if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size() ||
        root < 0 || root >= comm->size()) {
      LOG("Invalid rank, root, or world size for native CCL `Gather`.");
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
      LOG("Per-rank byte size overflows `size_t` for native CCL `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t rank_bytes = count * type_size;
    const size_t world_size = static_cast<size_t>(comm->size());
    if (rank_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Total byte size overflows `size_t` for native CCL `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }

    auto native_stream = reinterpret_cast<typename Api::Stream>(stream);
    ReturnStatus status = Api::Check(Api::GroupStart());
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    ReturnStatus first_error = Api::Check(Api::Send(
        send_buff, count, native_type, root, instance->handle, native_stream));

    if (comm->rank() == root) {
      auto *recv_bytes = static_cast<std::byte *>(recv_buff);
      for (int peer = 0; peer < comm->size(); ++peer) {
        const size_t offset = static_cast<size_t>(peer) * rank_bytes;
        status = Api::Check(Api::Recv(recv_bytes + offset, count, native_type,
                                      peer, instance->handle, native_stream));
        if (first_error == ReturnStatus::kSuccess &&
            status != ReturnStatus::kSuccess) {
          first_error = status;
        }
      }
    }

    const ReturnStatus group_end_status = Api::Check(Api::GroupEnd());
    return first_error != ReturnStatus::kSuccess ? first_error
                                                 : group_end_status;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_GATHER_H_
