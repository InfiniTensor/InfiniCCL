#include "devices/cpu/ibv_memory.h"

namespace infini::ccl {
namespace {

class HostMemoryAdapter : public IbvMemoryAdapter {
 public:
  ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) override {
    if (stream) {
      return ReturnStatus::kNotSupported;
    }
    key_ = IbvMemoryKey{reinterpret_cast<uintptr_t>(buffer), bytes, 0};
    return ReturnStatus::kSuccess;
  }

  ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                        IbvMemoryRegion *region) override {
    return RegisterIbvHostMemory(pd, buffer, bytes, region);
  }

  ReturnStatus Complete(bool /*receive*/) override {
    return ReturnStatus::kSuccess;
  }

  IbvMemoryKey memory_key() const override { return key_; }

 private:
  IbvMemoryKey key_;
};

}  // namespace

ReturnStatus MemoryAdapterFactory<Device::Type::kCpu>::Create(
    std::unique_ptr<IbvMemoryAdapter> *adapter) {
  if (!adapter) {
    return ReturnStatus::kInvalidArgument;
  }
  *adapter = std::make_unique<HostMemoryAdapter>();
  return ReturnStatus::kSuccess;
}

}  // namespace infini::ccl
