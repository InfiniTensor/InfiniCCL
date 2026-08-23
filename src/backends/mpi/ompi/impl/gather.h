#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GATHER_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GATHER_H_

#include <cstdlib>
#include <limits>
#include <memory>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/gather.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class GatherImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type, int root,
                            Communicator *comm, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Gather>{});
    using Rt = Runtime<kDev>;

    if (!comm || !comm->inter_comm() ||
        comm->inter_comm_backend() != BackendType::kOmpi) {
      LOG("Invalid OpenMPI communicator instance for `Gather`.");
      return ReturnStatus::kInternalError;
    }
    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator handle for `Gather`.");
      return ReturnStatus::kInternalError;
    }
    if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size()) {
      LOG("Invalid rank or world size for `Gather`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Byte size overflow for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t send_bytes = count * type_size;
    if (send_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("Per-rank byte count exceeds MPI int range for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t world_size = static_cast<size_t>(comm->size());
    if (send_bytes > std::numeric_limits<size_t>::max() / world_size) {
      LOG("Total byte size overflows `size_t` for `Gather`.");
      return ReturnStatus::kInvalidArgument;
    }
    const size_t recv_bytes = send_bytes * world_size;
    const bool is_root = comm->rank() == root;
    int mpi_byte_count = static_cast<int>(send_bytes);

    // Transfer raw bytes so movement-only collectives preserve every InfiniCCL
    // data type, including float16 and bfloat16.
    std::unique_ptr<void, decltype(&std::free)> host_sendbuf(
        std::malloc(send_bytes), &std::free);
    std::unique_ptr<void, decltype(&std::free)> host_recvbuf(
        is_root ? std::malloc(recv_bytes) : nullptr, &std::free);
    if (!host_sendbuf || (is_root && !host_recvbuf)) {
      LOG("Failed to allocate host buffers for `Gather` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf.get(), send_buff, send_bytes,
                                Rt::MemcpyDeviceToHost));
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    // Note: `MPI_Gather`'s `recvcount` is the per-rank count, not the total.
    INFINI_CHECK_MPI(MPI_Gather(host_sendbuf.get(), mpi_byte_count, MPI_BYTE,
                                host_recvbuf.get(), mpi_byte_count, MPI_BYTE,
                                root, inst->handle));

    if (is_root) {
      CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf.get(), recv_bytes,
                                  Rt::MemcpyHostToDevice));
    }

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Gather, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_GATHER_H_
