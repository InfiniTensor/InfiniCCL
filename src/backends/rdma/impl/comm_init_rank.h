#ifndef INFINI_CCL_BACKENDS_RDMA_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_RDMA_IMPL_COMM_INIT_RANK_H_

#include <cstring>
#include <memory>
#include <utility>

#include "backends/rdma/comm_instance.h"
#include "base/comm_init_rank.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class CommInitRankImpl<BackendType::kRdma, device_type> {
 public:
  static ReturnStatus Apply(Communicator *comm, int nranks,
                            infinicclUniqueId id, int rank) {
    if (!comm || rank < 0 || rank >= nranks) {
      return ReturnStatus::kInvalidArgument;
    }
    if (nranks != 2) {
      return ReturnStatus::kNotSupported;
    }
    if (comm->inter_comm()) {
      return ReturnStatus::kInvalidUsage;
    }

    RdmaUniqueId rdma_id{};
    std::memcpy(&rdma_id, id.internal, sizeof(rdma_id));
    if (rdma_id.port == 0 || rdma_id.address[0] == '\0' ||
        !std::memchr(rdma_id.address, '\0', sizeof(rdma_id.address))) {
      LOG("Invalid RDMA bootstrap ID for `CommInitRank`.");
      return ReturnStatus::kInvalidArgument;
    }
    auto instance = std::make_unique<RdmaInstance>();
    const auto status = instance->Connect(rdma_id, rank);
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    comm->set_world_info(rank, nranks);
    comm->set_local_size(1);
    comm->set_inter_comm(std::move(instance));
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommInitRank, BackendType::kRdma> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_IMPL_COMM_INIT_RANK_H_
