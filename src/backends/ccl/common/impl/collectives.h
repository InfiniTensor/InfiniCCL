#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COLLECTIVES_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COLLECTIVES_H_

#include "backends/ccl/common/api.h"
#include "device.h"
#include "traits.h"
#include "data_type_impl.h"
#include "return_status_impl.h"
#include "base/all_gather.h"
#include "base/all_reduce.h"
#include "base/all_to_all.h"
#include "base/broadcast.h"
#include "base/gather.h"
#include "base/reduce.h"
#include "base/reduce_scatter.h"
#include "base/scatter.h"
#include "backends/ccl/common/comm_instance.h"
#include "communicator.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
struct CclCollective {
  using Api = CclApi<backend, device>;
  using TypeMap = CclTypeMap<backend, device>;
  using CommInstance = CclCommInstance<Api>;

  static CommInstance *Get(Communicator *comm) {
    auto *instance = static_cast<CommInstance *>(comm->intra_comm());
    return instance && instance->handle ? instance : nullptr;
  }

  static bool ToDataType(DataType value, typename Api::DataType *result) {
    return TypeMap::ToBackendDataType(value, result);
  }

  static bool ToReduction(ReductionOpType value, typename Api::RedOp *result) {
    return TypeMap::ToBackendRedOp(value, result);
  }
};

template <BackendType backend, Device::Type device>
struct CclBroadcastImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, int root, Communicator *comm,
                            void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type)) return ReturnStatus::kNotSupported;
    return C::Api::Check(C::Api::Broadcast(send_buff, recv_buff, count,
                                            backend_type, root, instance->handle,
                                            reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

template <BackendType backend, Device::Type device>
struct CclReduceImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, ReductionOpType op, int root,
                            Communicator *comm, void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    typename C::Api::RedOp backend_op{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type) || !C::ToReduction(op, &backend_op)) {
      return ReturnStatus::kNotSupported;
    }
    return C::Api::Check(C::Api::Reduce(send_buff, recv_buff, count, backend_type,
                                        backend_op, root, instance->handle,
                                        reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

template <BackendType backend, Device::Type device>
struct CclReduceScatterImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, ReductionOpType op, Communicator *comm,
                            void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    typename C::Api::RedOp backend_op{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type) || !C::ToReduction(op, &backend_op)) {
      return ReturnStatus::kNotSupported;
    }
    return C::Api::Check(C::Api::ReduceScatter(send_buff, recv_buff, count,
                                               backend_type, backend_op,
                                               instance->handle,
                                               reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

template <BackendType backend, Device::Type device>
struct CclAllGatherImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, Communicator *comm, void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type)) return ReturnStatus::kNotSupported;
    return C::Api::Check(C::Api::AllGather(send_buff, recv_buff, count,
                                            backend_type, instance->handle,
                                            reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

template <BackendType backend, Device::Type device>
struct CclGatherImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, int root, Communicator *comm,
                            void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type)) return ReturnStatus::kNotSupported;
    return C::Api::Check(C::Api::Gather(send_buff, recv_buff, count, backend_type,
                                         root, instance->handle,
                                         reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

template <BackendType backend, Device::Type device>
struct CclScatterImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, int root, Communicator *comm,
                            void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type)) return ReturnStatus::kNotSupported;
    return C::Api::Check(C::Api::Scatter(send_buff, recv_buff, count, backend_type,
                                          root, instance->handle,
                                          reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

template <BackendType backend, Device::Type device>
struct CclAllToAllImpl {
  static ReturnStatus Apply(const void *send_buff, void *recv_buff, size_t count,
                            DataType type, Communicator *comm, void *stream) {
    using C = CclCollective<backend, device>;
    auto *instance = C::Get(comm);
    typename C::Api::DataType backend_type{};
    if (!instance) return ReturnStatus::kInternalError;
    if (!C::ToDataType(type, &backend_type)) return ReturnStatus::kNotSupported;
    return C::Api::Check(C::Api::AllToAll(send_buff, recv_buff, count,
                                           backend_type, instance->handle,
                                           reinterpret_cast<typename C::Api::Stream>(stream)));
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_COLLECTIVES_H_
