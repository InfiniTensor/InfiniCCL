#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_MARS_API_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_MARS_API_H_

#include <hccl.h>

#include "backends/ccl/common/api.h"
#include "devices/mars/runtime_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kHccl, Device::Type::kMars> {
  static constexpr BackendType kBackendType = BackendType::kHccl;
  static constexpr Device::Type kDeviceType = Device::Type::kMars;

  using Comm = hcclComm_t;
  using UniqueId = hcclUniqueId;
  using Result = hcclResult_t;
  using DataType = hcclDataType_t;
  using RedOp = hcclRedOp_t;
  using Stream = Runtime<Device::Type::kMars>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != hcclSuccess) {
      LOG(hcclGetErrorString(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static constexpr auto GetUniqueId = hcclGetUniqueId;
  static constexpr auto CommInitRank = hcclCommInitRank;
  static constexpr auto CommDestroy = hcclCommDestroy;
  static constexpr auto AllReduce = hcclAllReduce;
  static constexpr auto AllGather = hcclAllGather;
  static constexpr auto ReduceScatter = hcclReduceScatter;
  static constexpr auto Send = hcclSend;
  static constexpr auto Recv = hcclRecv;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_MARS_API_H_
