#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_TO_ALL_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_TO_ALL_H_

#include <cstdlib>
#include <limits>
#include <memory>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/all_to_all.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "dispatcher.h"
#include "logging.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class AllToAllImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllToAll>{});
    using Rt = Runtime<kDev>;

    if (!comm || !comm->inter_comm() ||
        comm->inter_comm_backend() != BackendType::kOmpi) {
      LOG("Invalid OpenMPI communicator instance for `AllToAll`.");
      return ReturnStatus::kInternalError;
    }
    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator handle for `AllToAll`.");
      return ReturnStatus::kInternalError;
    }
    if (comm->size() <= 0) {
      LOG("Invalid world size for `AllToAll`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Per-peer byte size overflows `size_t` for `AllToAll`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t peer_bytes = count * type_size;
    if (peer_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("Per-peer byte count exceeds MPI `int` range for `AllToAll`.");
      return ReturnStatus::kInvalidArgument;
    }

    size_t world_size = static_cast<size_t>(comm->size());
    if (peer_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Total byte size overflows `size_t` for `AllToAll`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t total_bytes = peer_bytes * world_size;
    int mpi_peer_bytes = static_cast<int>(peer_bytes);

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    std::unique_ptr<void, decltype(&std::free)> host_sendbuf(
        std::malloc(total_bytes), &std::free);
    std::unique_ptr<void, decltype(&std::free)> host_recvbuf(
        std::malloc(total_bytes), &std::free);
    if (!host_sendbuf || !host_recvbuf) {
      LOG("Failed to allocate host buffers for `AllToAll` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf.get(), send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));

    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    INFINI_CHECK_MPI(MPI_Alltoall(host_sendbuf.get(), mpi_peer_bytes, MPI_BYTE,
                                  host_recvbuf.get(), mpi_peer_bytes, MPI_BYTE,
                                  inst->handle));

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf.get(), total_bytes,
                                Rt::MemcpyHostToDevice));

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllToAll, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_ALL_TO_ALL_H_
