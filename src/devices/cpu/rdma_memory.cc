#include "devices/cpu/rdma_memory.h"

namespace infini::ccl {
namespace {

class HostMemoryAdapter : public RdmaMemoryAdapter {
 public:
  ReturnStatus Prepare(void * /*buffer*/, size_t /*bytes*/,
                       void *stream) override {
    return stream ? ReturnStatus::kNotSupported : ReturnStatus::kSuccess;
  }

  ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                        RdmaMemoryRegion *region) override {
    return RegisterRdmaHostMemory(pd, buffer, bytes, region);
  }

  ReturnStatus Complete(bool /*receive*/) override {
    return ReturnStatus::kSuccess;
  }
};

}  // namespace

ReturnStatus MemoryAdapterFactory<Device::Type::kCpu>::Create(
    std::unique_ptr<RdmaMemoryAdapter> *adapter) {
  if (!adapter) {
    return ReturnStatus::kInvalidArgument;
  }
  *adapter = std::make_unique<HostMemoryAdapter>();
  return ReturnStatus::kSuccess;
}

}  // namespace infini::ccl
