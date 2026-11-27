#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_GATHER_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_GATHER_H_

#include <algorithm>
#include <limits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "backends/mpi/ompi/type_map.h"
#include "base/all_gather.h"
#include "communicator.h"
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

    MPI_Datatype mpi_type = DataTypeToOmpiType(data_type);
    size_t type_size = kDataTypeToSize.at(data_type);
    if (comm->size() <= 0 ||
        count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Invalid AllGather size or send byte size overflow.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t world_size = static_cast<size_t>(comm->size());
    const size_t send_bytes = count * type_size;
    if (send_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("AllGather receive byte size overflow.");
      return ReturnStatus::kInvalidArgument;
    }
    constexpr size_t kMaxMpiCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    const size_t chunk_capacity = std::min(count, kMaxMpiCount);
    const size_t recv_bytes = chunk_capacity * type_size * world_size;

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    void *host_sendbuf = malloc(send_bytes);
    void *host_recvbuf = malloc(recv_bytes);
    if (!host_sendbuf || !host_recvbuf) {
      free(host_sendbuf);
      free(host_recvbuf);
      LOG("Failed to allocate host buffers for `AllGather` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, send_bytes,
                                Rt::MemcpyDeviceToHost));

    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    size_t offset = 0;
    while (offset < count) {
      const size_t chunk = std::min(count - offset, kMaxMpiCount);
      const size_t chunk_bytes = chunk * type_size;
      INFINI_CHECK_MPI(MPI_Allgather(
          static_cast<const char *>(host_sendbuf) + offset * type_size,
          static_cast<int>(chunk), mpi_type, host_recvbuf,
          static_cast<int>(chunk), mpi_type, inst->handle));

      if (count <= kMaxMpiCount) {
        CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, recv_bytes,
                                    Rt::MemcpyHostToDevice));
      } else {
        // Each round gathers rank-local chunks. Restore the full rank-major
        // layout rather than concatenating the rounds.
        for (size_t rank = 0; rank < world_size; ++rank) {
          CHECK_STATUS(Rt,
                       Rt::Memcpy(static_cast<char *>(recv_buff) +
                                      rank * send_bytes + offset * type_size,
                                  static_cast<const char *>(host_recvbuf) +
                                      rank * chunk_bytes,
                                  chunk_bytes, Rt::MemcpyHostToDevice));
        }
      }
      offset += chunk;
    }

    free(host_sendbuf);
    free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllGather, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_GATHER_H_
