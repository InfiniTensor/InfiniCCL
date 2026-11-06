#ifndef INFINI_CCL_BACKENDS_RDMA_MEMORY_ADAPTOR_H_
#define INFINI_CCL_BACKENDS_RDMA_MEMORY_ADAPTOR_H_

#include <infiniband/verbs.h>

#include <cstddef>
#include <memory>

#include "device.h"
#include "return_status_impl.h"

namespace infini::ccl {

struct RdmaMemoryRegion {
  ~RdmaMemoryRegion();

  RdmaMemoryRegion() = default;

  RdmaMemoryRegion(const RdmaMemoryRegion &) = delete;

  RdmaMemoryRegion &operator=(const RdmaMemoryRegion &) = delete;

  ReturnStatus Reset();

  ibv_mr *mr = nullptr;

  int fd = -1;
};

class RdmaMemoryAdapter {
 public:
  virtual ~RdmaMemoryAdapter() = default;

  virtual ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) = 0;

  virtual ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                                RdmaMemoryRegion *region) = 0;

  virtual ReturnStatus Complete(bool receive) = 0;
};

template <Device::Type device>
struct MemoryAdapterFactory {
  static ReturnStatus Create(std::unique_ptr<RdmaMemoryAdapter> * /*adapter*/) {
    return ReturnStatus::kNotSupported;
  }
};

ReturnStatus RegisterRdmaHostMemory(ibv_pd *pd, void *buffer, size_t bytes,
                                    RdmaMemoryRegion *region);

ReturnStatus CreateRdmaMemoryAdapter(
    Device::Type device, std::unique_ptr<RdmaMemoryAdapter> *adapter);

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_MEMORY_ADAPTOR_H_
