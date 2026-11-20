#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SCATTER_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SCATTER_H_

#include <cstddef>
#include <limits>

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/scatter.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclScatterImpl {
 private:
  using Api = CclApi<backend, device>;
  using TypeMap = CclTypeMap<backend, device>;
  using CommInstance = CclCommInstance<Api>;

  struct Arguments {
    typename Api::Comm handle{};
    typename Api::DataType data_type{};
    typename Api::Stream stream{};
    int nranks = 0;
    int rank = 0;
    int root = 0;
    size_t rank_bytes = 0;
  };

  static ReturnStatus Prepare(const void *send_buff, void *recv_buff,
                              size_t count, DataType data_type, int root,
                              Communicator *comm, void *stream,
                              Arguments *args) {
    if (!comm || !comm->intra_comm() || comm->intra_comm_backend() != backend ||
        comm->device_type() != device) {
      return ReturnStatus::kInternalError;
    }

    if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size() ||
        root < 0 || root >= comm->size()) {
      LOG("Invalid rank, root, or world size for CCL `Scatter`.");
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
      LOG("Per-rank byte size overflows `size_t` for CCL `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t rank_bytes = count * type_size;
    const size_t world_size = static_cast<size_t>(comm->size());
    if (rank_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Total byte size overflows `size_t` for CCL `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }

    args->handle = instance->handle;
    args->data_type = native_type;
    args->stream = reinterpret_cast<typename Api::Stream>(stream);
    args->nranks = comm->size();
    args->rank = comm->rank();
    args->root = root;
    args->rank_bytes = rank_bytes;
    return ReturnStatus::kSuccess;
  }

  static ReturnStatus ApplyGrouped(const void *send_buff, void *recv_buff,
                                   size_t count, const Arguments &args) {
    ReturnStatus status = Api::Check(Api::GroupStart());
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    ReturnStatus first_error = ReturnStatus::kSuccess;
    const auto *send_bytes = static_cast<const char *>(send_buff);

    if (args.rank == args.root) {
      for (int peer = 0; peer < args.nranks; ++peer) {
        const size_t offset = static_cast<size_t>(peer) * args.rank_bytes;
        status =
            Api::Check(Api::Send(send_bytes + offset, count, args.data_type,
                                 peer, args.handle, args.stream));
        if (status != ReturnStatus::kSuccess &&
            first_error == ReturnStatus::kSuccess) {
          first_error = status;
        }
      }
    }

    status = Api::Check(Api::Recv(recv_buff, count, args.data_type, args.root,
                                  args.handle, args.stream));
    if (status != ReturnStatus::kSuccess &&
        first_error == ReturnStatus::kSuccess) {
      first_error = status;
    }

    // Always close a successfully opened group, even after an enqueue error.
    const ReturnStatus group_end_status = Api::Check(Api::GroupEnd());
    return first_error != ReturnStatus::kSuccess ? first_error
                                                 : group_end_status;
  }

 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    Arguments args;
    const ReturnStatus status = Prepare(send_buff, recv_buff, count, data_type,
                                        root, comm, stream, &args);
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    if constexpr (Api::kHasNativeScatter) {
      return Api::Scatter(send_buff, recv_buff, count, args.data_type,
                          args.root, args.handle, args.stream);
    } else {
      return ApplyGrouped(send_buff, recv_buff, count, args);
    }
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SCATTER_H_
