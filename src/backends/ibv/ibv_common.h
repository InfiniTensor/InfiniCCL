#ifndef INFINI_CCL_BACKENDS_IBV_IBV_COMMON_H_
#define INFINI_CCL_BACKENDS_IBV_IBV_COMMON_H_

#include <cstdint>

#include "comm.h"

#define INFINICCL_IBV_ADDRESS_BYTES 64
#define INFINICCL_IBV_GROUP_ID_BYTES 16

namespace infini::ccl {

static_assert(INFINICCL_UNIQUE_ID_BYTES >=
                  sizeof(std::uint16_t) + INFINICCL_IBV_ADDRESS_BYTES +
                      INFINICCL_IBV_GROUP_ID_BYTES + 1,
              "The unique ID must hold the IBV endpoint, group ID and "
              "reserved space.");

struct IbvUniqueId {
  std::uint16_t port;

  char address[INFINICCL_IBV_ADDRESS_BYTES];

  char group_id[INFINICCL_IBV_GROUP_ID_BYTES];

  char reserved[INFINICCL_UNIQUE_ID_BYTES - sizeof(port) - sizeof(address) -
                sizeof(group_id)];
};

static_assert(sizeof(IbvUniqueId) == INFINICCL_UNIQUE_ID_BYTES,
              "`IbvUniqueId` must fit in `infinicclUniqueId::internal`.");

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_IBV_COMMON_H_
