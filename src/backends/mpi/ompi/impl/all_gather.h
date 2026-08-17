#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_GATHER_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_GATHER_H_

#include <cstdlib>
#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/all_gather.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "dispatcher.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class AllGatherImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllGather>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());

    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for `AllGather`.");
      return ReturnStatus::kInternalError;
    }

    if (comm->size() <= 0) {
      LOG("Invalid world size for `AllGather`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Byte size overflow for `AllGather`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t send_bytes = count * type_size;
    if (send_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("Per-rank byte count exceeds MPI int range for `AllGather`.");
      return ReturnStatus::kInvalidArgument;
    }

    size_t world_size = static_cast<size_t>(comm->size());
    if (world_size != 0 &&
        send_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Receive byte size overflow for `AllGather`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t recv_bytes = send_bytes * world_size;
    int mpi_byte_count = static_cast<int>(send_bytes);

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    void *host_sendbuf = std::malloc(send_bytes == 0 ? 1 : send_bytes);
    void *host_recvbuf = std::malloc(recv_bytes == 0 ? 1 : recv_bytes);
    if (!host_sendbuf || !host_recvbuf) {
      std::free(host_sendbuf);
      std::free(host_recvbuf);
      LOG("Failed to allocate host buffers for `AllGather` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, send_bytes,
                                Rt::MemcpyDeviceToHost));

    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    INFINI_CHECK_MPI(MPI_Allgather(host_sendbuf, mpi_byte_count, MPI_BYTE,
                                   host_recvbuf, mpi_byte_count, MPI_BYTE,
                                   inst->handle));

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, recv_bytes,
                                Rt::MemcpyHostToDevice));

    std::free(host_sendbuf);
    std::free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllGather, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_GATHER_H_
