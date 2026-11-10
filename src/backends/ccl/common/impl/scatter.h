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
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;

    if (!comm || !comm->intra_comm() || comm->intra_comm_backend() != backend ||
        comm->device_type() != device) {
      return ReturnStatus::kInternalError;
    }

    if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size() ||
        root < 0 || root >= comm->size()) {
      LOG("Invalid rank, root, or world size for native CCL `Scatter`.");
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
      LOG("Per-rank byte size overflows `size_t` for native CCL `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t rank_bytes = count * type_size;
    const size_t world_size = static_cast<size_t>(comm->size());
    if (rank_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Total byte size overflows `size_t` for native CCL `Scatter`.");
      return ReturnStatus::kInvalidArgument;
    }

    return Api::Check(Api::Scatter(
        send_buff, recv_buff, count, native_type, root, instance->handle,
        reinterpret_cast<typename Api::Stream>(stream)));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_SCATTER_H_
