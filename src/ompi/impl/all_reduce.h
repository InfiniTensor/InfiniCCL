#ifndef INFINI_CCL_OMPI_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_OMPI_IMPL_ALL_REDUCE_H_

#include "base/all_reduce.h"
#include "communicator.h"
#include "dispatcher.h"
#include "logging.h"
#include "ompi/checks.h"
#include "ompi/comm_instance.h"
#include "ompi/type_map.h"

namespace infini::ccl {

template <Device::Type device_type>
class AllReduceImpl<BackendType::kOmpi, device_type> {
public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            ReductionOpType op, Communicator *comm,
                            void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllReduce>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());

    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for AllReduce.");
      return ReturnStatus::kInternalError;
    }

    MPI_Datatype mpi_type = DataTypeToOmpiType(data_type);
    MPI_Op mpi_op = RedOpToOmpiOp(op);
    size_t type_size = kDataTypeToSize.at(data_type);
    size_t total_bytes = count * type_size;

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    void *host_sendbuf = malloc(total_bytes);
    void *host_recvbuf = malloc(total_bytes);
    if (!host_sendbuf || !host_recvbuf) {
      free(host_sendbuf);
      free(host_recvbuf);
      LOG("Failed to allocate host buffers for AllReduce staging.");
      return ReturnStatus::kSystemError;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));

    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    INFINI_CHECK_MPI(MPI_Allreduce(host_sendbuf, host_recvbuf, count, mpi_type,
                                   mpi_op, inst->handle));

    if (op == ReductionOpType::kAvg) {
      int size = comm->size();
      float scale = 1.0f / static_cast<float>(size);

      DispatchFunc<kDev, AllTypes>(data_type, [&](auto dtype) {
        using T = typename decltype(dtype)::type;

        T *typed_buf = static_cast<T *>(host_recvbuf);

        // Simply do the averaging on the CPU before the H2D copy.
        for (size_t i = 0; i < count; ++i) {
          // TODO(lzm): should later use the unified `Cast` function instead of
          // static_cast to support CPU custom types.
          typed_buf[i] *= static_cast<T>(scale);
        }
      });
    }

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, total_bytes,
                                Rt::MemcpyHostToDevice));

    free(host_sendbuf);
    free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllReduce, BackendType::kOmpi> : std::true_type {};

} // namespace infini::ccl

#endif // INFINI_CCL_OMPI_IMPL_ALL_REDUCE_H_
