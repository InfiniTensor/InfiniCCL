#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_API_H_

#include <cstddef>
#include <string>

#include "backends/ccl/common/api.h"
#include "backends/ccl/bkcl/checks.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct BkclApi {
  static constexpr BackendType kBackendType = BackendType::kBkcl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = XCCLContext_t;
  using UniqueId = XCCLUniqueId;
  using Result = XCCLResult_t;
  using DataType = XCCLDataType;
  using RedOp = XCCLOp;
  using Stream = typename Runtime<device>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != XCCL_SUCCESS) {
      LOG(("BKCL/XCCL error code: " + std::to_string(static_cast<int>(result)))
              .c_str());
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId* id) { return xccl_get_unique_id(id); }

  static Result CommInitRank(Comm* comm, int nranks, const UniqueId& id,
                             int rank) {
    return xccl_init_rank(comm, rank, nranks, &id);
  }

  static Result CommDestroy(Comm comm) { return xccl_destroy_context(comm); }

  static Result AllReduce(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return xccl_all_reduce(comm, send_buff, recv_buff, count, data_type, op,
                           stream);
  }

  static Result AllGather(const void* send_buff, void* recv_buff,
                          size_t send_count, DataType data_type, Comm comm,
                          Stream stream) {
    return xccl_all_gather(comm, send_buff, send_count, recv_buff, data_type,
                           stream);
  }

  static Result ReduceScatter(const void* send_buff, void* recv_buff,
                              size_t recv_count, DataType data_type, RedOp op,
                              Comm comm, Stream stream) {
    return xccl_reduce_scatter(comm, send_buff, recv_buff, recv_count,
                               data_type, op, stream);
  }

  static Result Send(const void* send_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return xccl_send(comm, send_buff, count, peer, data_type, stream);
  }

  static Result Recv(void* recv_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return xccl_recv(comm, recv_buff, count, peer, data_type, stream);
  }

  static Result Broadcast(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, int root, Comm comm,
                          Stream stream) {
    return xccl_broadcast(comm, send_buff, recv_buff, count, data_type, root,
                          stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_API_H_
