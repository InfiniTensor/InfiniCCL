#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_API_H_

#include <hccl/hccl.h>

#include <cstddef>
#include <cstdint>

#include "backends/ccl/common/api.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct HcclApi {
  static constexpr BackendType kBackendType = BackendType::kHccl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = HcclComm;
  using Result = HcclResult;
  using DataType = HcclDataType;
  using RedOp = HcclReduceOp;
  using Stream = typename Runtime<device>::Stream;

  struct ThreadLocalStream {
    Stream stream{};
    Result status = HCCL_SUCCESS;

    ThreadLocalStream() {
      if (Runtime<device>::StreamCreate(&stream) != 0) {
        status = HCCL_E_RUNTIME;
      }
    }

    ~ThreadLocalStream() {
      if (stream != nullptr) {
        Runtime<device>::StreamDestroy(stream);
      }
    }
  };

  static ReturnStatus Check(Result result) {
    if (result != HCCL_SUCCESS) {
      const char* message = HcclGetErrorString(result);
      LOG(message ? message : "Unknown HCCL error");
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result CommInitAll(Comm* comms, int n_dev, const int* dev_list,
                            const int*) {
    return HcclCommInitAll(
        static_cast<uint32_t>(n_dev),
        reinterpret_cast<int32_t*>(const_cast<int*>(dev_list)), comms);
  }

  static Result CommDestroy(Comm comm) { return HcclCommDestroy(comm); }

  static Result ResolveStream(Stream requested, Stream* resolved) {
    if (requested != nullptr) {
      *resolved = requested;
      return HCCL_SUCCESS;
    }

    thread_local ThreadLocalStream default_stream;
    if (default_stream.status != HCCL_SUCCESS) {
      return default_stream.status;
    }
    *resolved = default_stream.stream;
    return HCCL_SUCCESS;
  }

  static Result AllReduce(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    auto status = ResolveStream(stream, &stream);
    if (status != HCCL_SUCCESS) {
      return status;
    }
    return HcclAllReduce(const_cast<void*>(send_buff), recv_buff, count,
                         data_type, op, comm, stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_API_H_
