#ifndef INFINI_CCL_NVIDIA_NCCL_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_NVIDIA_NCCL_IMPL_COMM_INIT_RANK_H_

#include <nccl.h>

#include "base/comm_init_rank.h"
#include "communicator.h"
#include "logging.h"
#include "nvidia/nccl/checks.h"
#include "nvidia/nccl/comm_instance.h"

namespace infini::ccl {

template <>
class CommInitRankImpl<BackendType::kNccl, Device::Type::kNvidia> {
 public:
  static ReturnStatus Apply(Communicator *comm, int nranks,
                            infinicclUniqueId id, int rank) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<CommInitRank>{});
    using Rt = Runtime<kDev>;

    const ncclUniqueId *nccl_id =
        reinterpret_cast<const ncclUniqueId *>(id.internal);

    ncclComm_t nccl_handle;
    INFINI_CHECK_NCCL(ncclCommInitRank(&nccl_handle, nranks, *nccl_id, rank));

    comm->set_world_info(rank, nranks);

    auto inst = std::make_unique<NcclInstance>();
    inst->handle = nccl_handle;

    comm->set_intra_comm(std::move(inst));

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommInitRank, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif
