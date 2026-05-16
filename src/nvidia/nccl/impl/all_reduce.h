#ifndef INFINI_CCL_NVIDIA_NCCL_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_NVIDIA_NCCL_IMPL_ALL_REDUCE_H_

#include "base/all_reduce.h"
#include "communicator.h"
#include "nvidia/nccl/checks.h"
#include "nvidia/nccl/comm_instance.h"
#include "nvidia/nccl/type_map.h"
#include "nvidia/runtime_.h"

namespace infini::ccl {

template <>
class AllReduceImpl<BackendType::kNccl, Device::Type::kNvidia> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            ReductionOpType op, Communicator *comm,
                            void *stream) {
    auto *comm_internal = static_cast<Communicator *>(comm);
    if (!comm_internal) {
      return ReturnStatus::kInternalError;
    }

    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllReduce>{});
    using Rt = Runtime<kDev>;

    auto *intra = static_cast<NcclInstance *>(comm_internal->intra_comm());
    if (!intra || !intra->handle) {
      return ReturnStatus::kInternalError;
    }

    ncclDataType_t nccl_type = DataTypeToNcclType(data_type);
    ncclRedOp_t nccl_op = RedOpToNcclOp(op);

    INFINI_CHECK_NCCL(ncclAllReduce(send_buff, recv_buff, count, nccl_type,
                                    nccl_op, intra->handle,
                                    reinterpret_cast<Rt::Stream>(stream)));

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllReduce, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_NVIDIA_NCCL_IMPL_ALL_REDUCE_H_
