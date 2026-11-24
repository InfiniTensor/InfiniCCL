#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_API_H_

#include <mccl.h>

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "backends/ccl/mccl/capabilities.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct McclDataTypeTraits;

template <Device::Type device>
struct McclApi {
  static constexpr BackendType kBackendType = BackendType::kMccl;
  static constexpr Device::Type kDeviceType = device;

  static constexpr bool kHasNativeAllToAll =
      mccl::capabilities::kHasNativeAllToAll<device>;
  static constexpr bool kHasNativeScatter =
      mccl::capabilities::kHasNativeScatter<device>;
  static constexpr bool kHasNativeGather =
      mccl::capabilities::kHasNativeGather<device>;

  using Comm = mcclComm_t;
  using UniqueId = mcclUniqueId;
  using Result = mcclResult_t;
  using DataType = mcclDataType_t;
  using RedOp = mcclRedOp_t;
  using Stream = typename Runtime<device>::Stream;

  static ReturnStatus Check(Result result) {
    if (result != mcclSuccess) {
      LOG(mcclGetErrorString(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId *id) { return mcclGetUniqueId(id); }

  static Result CommInitRank(Comm *comm, int nranks, UniqueId id, int rank) {
    return mcclCommInitRank(comm, nranks, id, rank);
  }

  static Result CommDestroy(Comm comm) { return mcclCommDestroy(comm); }

  static ReturnStatus Scatter(const void *send_buff, void *recv_buff,
                              size_t count, DataType data_type, int root,
                              Comm comm, Stream stream) {
    if constexpr (mccl::capabilities::kHasScatter<device>) {
      return Check(mcclScatter(send_buff, recv_buff, count, data_type, root,
                               comm, stream));
    } else {
      LOG("Native MCCL `Scatter` is unavailable.");
      return ReturnStatus::kNotSupported;
    }
  }

  static Result Broadcast(const void *send_buff, void *recv_buff, size_t count,
                          DataType data_type, int root, Comm comm,
                          Stream stream) {
    return mcclBroadcast(send_buff, recv_buff, count, data_type, root, comm,
                         stream);
  }

  static Result AllReduce(const void *send_buff, void *recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return mcclAllReduce(send_buff, recv_buff, count, data_type, op, comm,
                         stream);
  }

  static ReturnStatus Gather(const void *send_buff, void *recv_buff,
                             size_t count, DataType data_type, int root,
                             Comm comm, Stream stream) {
    if constexpr (mccl::capabilities::kHasGather<device>) {
      return Check(mcclGather(send_buff, recv_buff, count, data_type, root,
                              comm, stream));
    } else {
      LOG("Native MCCL `Gather` is unavailable.");
      return ReturnStatus::kNotSupported;
    }
  }

  static Result AllGather(const void *send_buff, void *recv_buff,
                          size_t send_count, DataType data_type, Comm comm,
                          Stream stream) {
    return mcclAllGather(send_buff, recv_buff, send_count, data_type, comm,
                         stream);
  }

  static ReturnStatus AllToAll(const void *send_buff, void *recv_buff,
                               size_t count, DataType data_type, Comm comm,
                               Stream stream) {
    if constexpr (mccl::capabilities::kHasAllToAll<device>) {
      return Check(
          mcclAllToAll(send_buff, recv_buff, count, data_type, comm, stream));
    } else if constexpr (mccl::capabilities::kHasAllToAllAlternate<device>) {
      return Check(
          mcclAlltoAll(send_buff, recv_buff, count, data_type, comm, stream));
    } else {
      LOG("Native MCCL `AllToAll` is unavailable.");
      return ReturnStatus::kNotSupported;
    }
  }

  static Result ReduceScatter(const void *send_buff, void *recv_buff,
                              size_t recv_count, DataType data_type, RedOp op,
                              Comm comm, Stream stream) {
    return mcclReduceScatter(send_buff, recv_buff, recv_count, data_type, op,
                             comm, stream);
  }

  static Result Send(const void *send_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return mcclSend(send_buff, count, data_type, peer, comm, stream);
  }

  static Result Recv(void *recv_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return mcclRecv(recv_buff, count, data_type, peer, comm, stream);
  }

  static Result GroupStart() { return mcclGroupStart(); }

  static Result GroupEnd() { return mcclGroupEnd(); }

  static Result Reduce(const void *send_buff, void *recv_buff, size_t count,
                       DataType data_type, RedOp op, int root, Comm comm,
                       Stream stream) {
    return mcclReduce(send_buff, recv_buff, count, data_type, op, root, comm,
                      stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_API_H_
