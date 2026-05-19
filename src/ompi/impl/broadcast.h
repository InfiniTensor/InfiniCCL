#ifndef INFINI_CCL_OMPI_IMPL_BROADCAST_H_
#define INFINI_CCL_OMPI_IMPL_BROADCAST_H_

#include <cstdlib>
#include <limits>

#include "base/broadcast.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "ompi/checks.h"
#include "ompi/comm_instance.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class BroadcastImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Broadcast>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for Broadcast.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Broadcast byte size overflow.");
      return ReturnStatus::kInvalidArgument;
    }

    size_t total_bytes = count * type_size;
    void *host_buf = std::malloc(total_bytes);
    if (!host_buf) {
      LOG("Failed to allocate host buffer for Broadcast staging.");
      return ReturnStatus::kSystemError;
    }

    if (comm->rank() == root) {
      CHECK_STATUS(Rt, Rt::Memcpy(host_buf, send_buff, total_bytes,
                                  Rt::MemcpyDeviceToHost));
      CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));
    }

    auto *bytes = static_cast<char *>(host_buf);
    size_t offset = 0;
    constexpr size_t kMaxMpiCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    while (offset < total_bytes) {
      size_t chunk = total_bytes - offset;
      if (chunk > kMaxMpiCount) {
        chunk = kMaxMpiCount;
      }
      INFINI_CHECK_MPI(MPI_Bcast(bytes + offset, static_cast<int>(chunk),
                                 MPI_BYTE, root, inst->handle));
      offset += chunk;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_buf, total_bytes,
                                Rt::MemcpyHostToDevice));

    std::free(host_buf);
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Broadcast, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_OMPI_IMPL_BROADCAST_H_
