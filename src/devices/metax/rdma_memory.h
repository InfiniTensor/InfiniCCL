#ifndef INFINI_CCL_DEVICES_METAX_RDMA_MEMORY_H_
#define INFINI_CCL_DEVICES_METAX_RDMA_MEMORY_H_

#include <memory>

#include "backends/rdma/memory_adaptor.h"

namespace infini::ccl {

template <>
struct MemoryAdapterFactory<Device::Type::kMetax> {
  static ReturnStatus Create(std::unique_ptr<RdmaMemoryAdapter> *adapter);
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_METAX_RDMA_MEMORY_H_
