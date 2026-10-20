#ifndef INFINI_CCL_DEVICES_NVIDIA_IBV_MEMORY_H_
#define INFINI_CCL_DEVICES_NVIDIA_IBV_MEMORY_H_

#include <memory>

#include "backends/ibv/memory_adaptor.h"

namespace infini::ccl {

template <>
struct MemoryAdapterFactory<Device::Type::kNvidia> {
  static ReturnStatus Create(std::unique_ptr<IbvMemoryAdapter> *adapter);
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_NVIDIA_IBV_MEMORY_H_
