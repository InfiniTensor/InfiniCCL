#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_API_H_

#include <hccl/hccl.h>

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "devices/ascend/checks.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct HcclApi {
  static constexpr BackendType kBackendType = BackendType::kHccl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = HcclComm;
  using UniqueId = HcclRootInfo;
  using Result = HcclResult;
  using DataType = HcclDataType;
  using RedOp = HcclReduceOp;
  using Stream = typename Runtime<device>::Stream;

 private:
  // HCCL requires a real stream. Keep one per thread, and recreate it when
  // the thread switches devices.
  struct DefaultStream {
    Stream stream = nullptr;

    int device_id = -1;

    void Destroy() {
      if (stream != nullptr) {
        int current_device = 0;
        INFINI_CHECK_ACL(aclrtGetDevice(&current_device));
        INFINI_CHECK_ACL(aclrtSetDevice(device_id));
        INFINI_CHECK_ACL(aclrtDestroyStream(stream));
        INFINI_CHECK_ACL(aclrtSetDevice(current_device));
        stream = nullptr;
      }
    }

    ~DefaultStream() { Destroy(); }
  };

  static Stream GetDefaultStream() {
    thread_local DefaultStream default_stream;

    int device_id = 0;
    INFINI_CHECK_ACL(aclrtGetDevice(&device_id));

    if (default_stream.device_id != device_id) {
      default_stream.Destroy();
      INFINI_CHECK_ACL(aclrtCreateStream(&default_stream.stream));
      default_stream.device_id = device_id;
    }
    return default_stream.stream;
  }

  template <typename Operation>
  static Result WithStream(Stream stream, Operation operation) {
    const bool use_default_stream = stream == nullptr;
    if (use_default_stream) {
      stream = GetDefaultStream();
    }

    Result result = operation(stream);
    // Preserve synchronous null-stream behavior; explicit streams stay async.
    if (use_default_stream && result == HCCL_SUCCESS) {
      INFINI_CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    return result;
  }

 public:
  static ReturnStatus Check(Result result) {
    if (result != HCCL_SUCCESS) {
      LOG(HcclGetErrorString(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId* id) {
    INFINI_CHECK_ACL(Runtime<device>::Initialize());
    // HCCL root-info creation requires a current device.
    int device_id = 0;
    if (Runtime<device>::GetDevice(&device_id) != ACL_SUCCESS) {
      INFINI_CHECK_ACL(Runtime<device>::SetDevice(0));
    }
    return HcclGetRootInfo(id);
  }

  static Result CommInitRank(Comm* comm, int nranks, UniqueId id, int rank) {
    return HcclCommInitRootInfo(nranks, &id, rank, comm);
  }

  static Result CommDestroy(Comm comm) { return HcclCommDestroy(comm); }

  static Result AllReduce(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return WithStream(stream, [&](Stream resolved) {
      return HcclAllReduce(const_cast<void*>(send_buff), recv_buff, count,
                           data_type, op, comm, resolved);
    });
  }

  static Result AllGather(const void* send_buff, void* recv_buff,
                          size_t send_count, DataType data_type, Comm comm,
                          Stream stream) {
    return WithStream(stream, [&](Stream resolved) {
      return HcclAllGather(const_cast<void*>(send_buff), recv_buff, send_count,
                           data_type, comm, resolved);
    });
  }

  static Result ReduceScatter(const void* send_buff, void* recv_buff,
                              size_t recv_count, DataType data_type, RedOp op,
                              Comm comm, Stream stream) {
    return WithStream(stream, [&](Stream resolved) {
      return HcclReduceScatter(const_cast<void*>(send_buff), recv_buff,
                               recv_count, data_type, op, comm, resolved);
    });
  }

  static Result Send(const void* send_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return WithStream(stream, [&](Stream resolved) {
      return HcclSend(const_cast<void*>(send_buff), count, data_type, peer,
                      comm, resolved);
    });
  }

  static Result Recv(void* recv_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return WithStream(stream, [&](Stream resolved) {
      return HcclRecv(recv_buff, count, data_type, peer, comm, resolved);
    });
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_API_H_
