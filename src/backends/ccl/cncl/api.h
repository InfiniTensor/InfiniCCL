#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_API_H_

#include <cncl.h>

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct CnclApi {
  static constexpr BackendType kBackendType = BackendType::kCncl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = cnclComm_t;
  using Result = cnclResult_t;
  using DataType = cnclDataType_t;
  using RedOp = cnclReduceOp_t;
  using Stream = typename Runtime<device>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != CNCL_RET_SUCCESS) {
      LOG(cnclGetErrorStr(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result CommInitAll(Comm* comms, int n_dev, const int* dev_list,
                            const int* rank_list) {
    return cnclInitComms(comms, n_dev, dev_list, rank_list, n_dev, nullptr);
  }

  static Result CommDestroy(Comm comm) { return cnclFreeComm(comm); }

  static Result AllReduce(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return cnclAllReduce(send_buff, recv_buff, count, data_type, op, comm,
                         stream);
  }

  static Result AllGather(const void* send_buff, void* recv_buff,
                          size_t send_count, DataType data_type, Comm comm,
                          Stream stream) {
    return cnclAllGather(send_buff, recv_buff, send_count, data_type, comm,
                         stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_API_H_
