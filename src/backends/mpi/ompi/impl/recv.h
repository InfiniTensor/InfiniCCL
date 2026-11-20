#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_RECV_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_RECV_H_

#include <limits>

#include "backends/mpi/ompi/buffer_plan.h"
#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "base/recv.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

template <Device::Type device_type>
class RecvImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(void *recv_buff, size_t count, DataType data_type,
                            int peer, Communicator *comm, void *stream) {
    constexpr Device::Type kDev = device_type;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid `OpenMPI` communicator instance for `Recv`.");
      return ReturnStatus::kInternalError;
    }

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Byte size overflow for `Recv`.");
      return ReturnStatus::kInvalidArgument;
    }

    const size_t total_bytes = count * type_size;
    MpiBufferPlan<kDev> plan;
    ReturnStatus status =
        plan.Prepare(nullptr, recv_buff, 0, total_bytes,
                     MpiDeviceAwareUsable(inst->device_support), stream);
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    auto *bytes = static_cast<unsigned char *>(plan.recv_buff());
    size_t offset = 0;
    constexpr size_t kMaxMpiCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    constexpr int kTag = 0;
    while (offset < total_bytes) {
      size_t chunk = total_bytes - offset;
      if (chunk > kMaxMpiCount) {
        chunk = kMaxMpiCount;
      }

      INFINI_CHECK_MPI(MPI_Recv(bytes + offset, static_cast<int>(chunk),
                                MPI_BYTE, peer, kTag, inst->handle,
                                MPI_STATUS_IGNORE));
      offset += chunk;
    }

    return plan.Commit();
  }
};

template <>
struct BackendEnabled<Recv, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_RECV_H_
