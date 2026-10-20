#ifndef INFINI_CCL_BACKENDS_IBV_IMPL_GET_UNIQUE_ID_H_
#define INFINI_CCL_BACKENDS_IBV_IMPL_GET_UNIQUE_ID_H_

#include <sys/random.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "backends/ibv/ibv_common.h"
#include "base/get_unique_id.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class GetUniqueIdImpl<BackendType::kIbv, device_type> {
 public:
  static ReturnStatus Apply(infinicclUniqueId *id) {
    if (!id) {
      LOG("Invalid unique ID pointer for `GetUniqueId`.");
      return ReturnStatus::kInvalidArgument;
    }

    const char *address = std::getenv("INFINICCL_IBV_ADDR");
    if (!address || address[0] == '\0' ||
        std::strlen(address) >= INFINICCL_IBV_ADDRESS_BYTES) {
      LOG("`INFINICCL_IBV_ADDR` must contain a non-empty address that fits "
          "in the IBV bootstrap address field.");
      return ReturnStatus::kInvalidArgument;
    }

    const char *port_string = std::getenv("INFINICCL_IBV_PORT");
    const char *value = port_string ? port_string : "7471";
    char *end = nullptr;
    errno = 0;
    const unsigned long port = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || port == 0 || port > 65535) {
      LOG("`INFINICCL_IBV_PORT` must be between 1 and 65535.");
      return ReturnStatus::kInvalidArgument;
    }

    IbvUniqueId ibv_id{};
    ibv_id.port = static_cast<std::uint16_t>(port);
    size_t offset = 0;
    while (offset < sizeof(ibv_id.group_id)) {
      const ssize_t count = getrandom(ibv_id.group_id + offset,
                                      sizeof(ibv_id.group_id) - offset, 0);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        return ReturnStatus::kSystemError;
      }
      offset += static_cast<size_t>(count);
    }
    std::strncpy(ibv_id.address, address, sizeof(ibv_id.address) - 1);
    std::memset(id->internal, 0, sizeof(id->internal));
    std::memcpy(id->internal, &ibv_id, sizeof(ibv_id));

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<GetUniqueId, BackendType::kIbv> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_IMPL_GET_UNIQUE_ID_H_
