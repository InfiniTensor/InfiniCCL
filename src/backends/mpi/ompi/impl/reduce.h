#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_REDUCE_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_REDUCE_H_

#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>

#include "backends/mpi/ompi/checks.h"
#include "backends/mpi/ompi/comm_instance.h"
#include "backends/mpi/ompi/type_map.h"
#include "base/reduce.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "dispatcher.h"
#include "logging.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device_type>
class ReduceImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            ReductionOpType op, int root, Communicator *comm,
                            void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<Reduce>{});
    using Rt = Runtime<kDev>;

    if (!comm || comm->inter_comm_backend() != BackendType::kOmpi) {
      LOG("Invalid OpenMPI communicator for `Reduce`.");
      return ReturnStatus::kInternalError;
    }

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());
    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for `Reduce`.");
      return ReturnStatus::kInternalError;
    }
    if (comm->size() <= 0) {
      LOG("Invalid world size for `Reduce`.");
      return ReturnStatus::kInternalError;
    }

    MPI_Datatype mpi_type = DataTypeToOmpiType(data_type);
    MPI_Op mpi_op = RedOpToOmpiOp(op);
    if (mpi_type == MPI_BYTE) {
      LOG("Data type is not supported by OpenMPI reductions for `Reduce`.");
      return ReturnStatus::kNotSupported;
    }

    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      LOG("`count` exceeds MPI int range for `Reduce`.");
      return ReturnStatus::kInvalidArgument;
    }
    int mpi_count = static_cast<int>(count);

    size_t type_size = kDataTypeToSize.at(data_type);
    if (count > std::numeric_limits<size_t>::max() / type_size) {
      LOG("Buffer byte size overflows `size_t` for `Reduce`.");
      return ReturnStatus::kInvalidArgument;
    }
    size_t total_bytes = count * type_size;
    const bool is_root = comm->rank() == root;

    // Host staging buffers. Only `root` allocates the receive side, since
    // `MPI_Reduce` writes the output only on `root`.
    std::unique_ptr<void, decltype(&std::free)> host_sendbuf(
        std::malloc(total_bytes), &std::free);
    std::unique_ptr<void, decltype(&std::free)> host_recvbuf(
        is_root ? std::malloc(total_bytes) : nullptr, &std::free);
    if (!host_sendbuf || (is_root && !host_recvbuf)) {
      LOG("Failed to allocate host buffers for `Reduce` staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf.get(), send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    INFINI_CHECK_MPI(MPI_Reduce(host_sendbuf.get(), host_recvbuf.get(),
                                mpi_count, mpi_type, mpi_op, root,
                                inst->handle));

    if (is_root) {
      if (op == ReductionOpType::kAvg) {
        float scale = 1.0f / static_cast<float>(comm->size());

        DispatchFunc<kDev, AllTypes>(data_type, [&](auto dtype) {
          using T = typename decltype(dtype)::type;

          T *typed_buf = static_cast<T *>(host_recvbuf.get());

          // Simply do the averaging on the CPU before the H2D copy.
          for (size_t i = 0; i < count; ++i) {
            if constexpr (std::is_integral_v<T>) {
              // Scale in floating point first; casting `scale` to an integer
              // type would truncate it to `0` and zero out the result.
              typed_buf[i] =
                  CastTo<kDev, T>(CastTo<kDev, double>(typed_buf[i]) * scale);
            } else if constexpr (SupportsOpValue<T, decltype(scale),
                                                 MulAssignOp>) {
              typed_buf[i] *= scale;
            } else {
              float f_val = ToFloat<kDev>(typed_buf[i]) * scale;
              typed_buf[i] = CastTo<kDev, T>(f_val);
            }
          }
        });
      }

      CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf.get(), total_bytes,
                                  Rt::MemcpyHostToDevice));
    }

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Reduce, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_IMPL_REDUCE_H_
