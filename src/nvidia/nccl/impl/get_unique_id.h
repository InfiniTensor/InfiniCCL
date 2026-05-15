#ifndef INFINI_CCL_NVIDIA_NCCL_IMPL_GET_UNIQUE_ID_H_
#define INFINI_CCL_NVIDIA_NCCL_IMPL_GET_UNIQUE_ID_H_

#include <cstring>

#include <nccl.h>

#include "base/get_unique_id.h"
#include "nvidia/nccl/checks.h"

namespace infini::ccl {

template <> class GetUniqueIdImpl<BackendType::kNccl, Device::Type::kNvidia> {
public:
  static ReturnStatus Apply(infiniUniqueId *id) {
    ncclUniqueId nccl_id;

    INFINI_CHECK_NCCL(ncclGetUniqueId(&nccl_id));

    // Safety check: ensure our public struct is large enough.
    static_assert(sizeof(ncclUniqueId) <= sizeof(id->internal),
                  "`infiniUniqueId::internal` is too small for `ncclUniqueId`");

    std::memcpy(id->internal, &nccl_id, sizeof(nccl_id));

    return ReturnStatus::kSuccess;
  }
};

// Enable this backend for the GetUniqueId operation.
template <>
struct BackendEnabled<GetUniqueId, BackendType::kNccl> : std::true_type {};

} // namespace infini::ccl

#endif // INFINI_CCL_NVIDIA_NCCL_IMPL_GET_UNIQUE_ID_H_