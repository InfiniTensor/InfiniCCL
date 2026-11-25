#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_BROADCAST_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_BROADCAST_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/broadcast.h"
#include "communicator.h"

namespace infini::ccl {

// The common impls have no broadcast (NCCL relies on `Bcast` -> in-place
// `Broadcast` only), so BKCL provides one directly. XCCL reads the payload
// from `send_buff` on the root rank; every other rank must pass NULL there
// (`xccl_broadcast` contract), which maps 1:1 onto the InfiniCCL signature.
template <Device::Type device>
class BroadcastImpl<BackendType::kBkcl, device> {
 public:
  static ReturnStatus Apply(const void* send_buff, void* recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator* comm, void* stream) {
    using Api = CclApi<BackendType::kBkcl, device>;
    using TypeMap = CclTypeMap<BackendType::kBkcl, device>;
    using CommInstance = CclCommInstance<Api>;

    auto* comm_internal = static_cast<Communicator*>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }

    auto* intra = static_cast<CommInstance*>(comm_internal->intra_comm());
    if (!intra || !intra->handle) {
      return ReturnStatus::kInternalError;
    }

    typename Api::DataType ccl_type{};
    if (!TypeMap::ToBackendDataType(data_type, &ccl_type)) {
      return ReturnStatus::kNotSupported;
    }

    const bool is_root = comm_internal->rank() == root;
    return Api::Check(Api::Broadcast(is_root ? send_buff : nullptr, recv_buff,
                                     count, ccl_type, root, intra->handle,
                                     reinterpret_cast<typename Api::Stream>(
                                         stream)));
  }
};

template <>
struct BackendEnabled<Broadcast, BackendType::kBkcl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_IMPL_BROADCAST_H_
