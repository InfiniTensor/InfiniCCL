#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_GATHER_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_GATHER_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/all_gather.h"
#include "communicator.h"

namespace infini::ccl {

template <BackendType>
struct DeferredAllGather {
  using type = AllGather;
};

template <BackendType backend, Device::Type device>
class CclAllGatherImpl {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
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

      using FallbackOperation = typename DeferredAllGather<backend>::type;
      if constexpr (BackendEnabled<FallbackOperation,
                                   BackendType::kOmpi>::value) {
        return AllGatherImpl<BackendType::kOmpi, device>::Apply(
            send_buff, recv_buff, count, data_type, comm, stream);
      }

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

    return Api::Check(Api::AllGather(
        send_buff, recv_buff, count, native_type, instance->handle,
        reinterpret_cast<typename Api::Stream>(stream)));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_GATHER_H_
