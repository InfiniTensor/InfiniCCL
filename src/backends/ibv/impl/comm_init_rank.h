#ifndef INFINI_CCL_BACKENDS_IBV_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_IBV_IMPL_COMM_INIT_RANK_H_

#include <cstring>
#include <memory>
#include <utility>

#include "backends/ibv/comm_instance.h"
#include "base/comm_init_rank.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class CommInitRankImpl<BackendType::kIbv, device_type> {
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

    IbvUniqueId ibv_id{};
    std::memcpy(&ibv_id, id.internal, sizeof(ibv_id));
    if (ibv_id.port == 0 || ibv_id.address[0] == '\0' ||
        !std::memchr(ibv_id.address, '\0', sizeof(ibv_id.address))) {
      LOG("Invalid IBV bootstrap ID for `CommInitRank`.");
      return ReturnStatus::kInvalidArgument;
    }
    auto instance = std::make_unique<IbvInstance>();
    const auto status = instance->Connect(ibv_id, rank);
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
struct BackendEnabled<CommInitRank, BackendType::kIbv> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_IMPL_COMM_INIT_RANK_H_
