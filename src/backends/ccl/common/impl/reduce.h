#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_REDUCE_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_REDUCE_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/reduce.h"
#include "communicator.h"
#include "logging.h"

namespace infini::ccl {

template <BackendType>
struct DeferredReduce {
  using type = Reduce;
};

template <BackendType backend, Device::Type device>
class CclReduceImpl {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            ReductionOpType op, int root, Communicator *comm,
                            void *stream) {
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

      using FallbackOperation = typename DeferredReduce<backend>::type;
      if constexpr (BackendEnabled<FallbackOperation,
                                   BackendType::kOmpi>::value) {
        return ReduceImpl<BackendType::kOmpi, device>::Apply(
            send_buff, recv_buff, count, data_type, op, root, comm, stream);
      }

      return ReturnStatus::kInternalError;
    }

    if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size() ||
        root < 0 || root >= comm->size()) {
      LOG("Invalid rank, root, or world size for native CCL `Reduce`.");
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

    typename Api::RedOp native_op{};
    if (!TypeMap::ToBackendRedOp(op, &native_op)) {
      return ReturnStatus::kNotSupported;
    }

    return Api::Check(Api::Reduce(
        send_buff, recv_buff, count, native_type, native_op, root,
        instance->handle, reinterpret_cast<typename Api::Stream>(stream)));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_REDUCE_H_
