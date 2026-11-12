#include "devices/cpu/rdma_memory.h"

namespace infini::ccl {
namespace {

class HostMemoryAdapter : public RdmaMemoryAdapter {
 public:
  ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) override {
    if (stream) {
      return ReturnStatus::kNotSupported;
    }
    key_ = RdmaMemoryKey{reinterpret_cast<uintptr_t>(buffer), bytes, 0};
    return ReturnStatus::kSuccess;
  }

  ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                        RdmaMemoryRegion *region) override {
    return RegisterRdmaHostMemory(pd, buffer, bytes, region);
  }

  ReturnStatus Complete(bool /*receive*/) override {
    return ReturnStatus::kSuccess;
  }

  RdmaMemoryKey memory_key() const override { return key_; }

  // Host memory never takes the RDMA path in the first iteration.
  bool CanRegister(const void * /*buffer*/, size_t /*bytes*/) const override {
    return false;
  }

 private:
  RdmaMemoryKey key_;
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
