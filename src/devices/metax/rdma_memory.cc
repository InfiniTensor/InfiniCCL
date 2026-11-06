#include "devices/metax/rdma_memory.h"

#include <mcr/mc_runtime.h>
#include <mxkw/mxkw.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <type_traits>

namespace infini::ccl {
namespace {

struct MetaxAllocation {
  void *base = nullptr;

  size_t bytes = 0;

  size_t offset = 0;
};

ReturnStatus CheckMaca(mcError_t status, const char *operation) {
  if (status != mcSuccess) {
    std::cerr << "MACA operation `" << operation
              << "` failed: " << mcGetErrorString(status) << ".\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

template <typename T>
uintptr_t PointerAddress(T pointer) {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<uintptr_t>(pointer);
  } else {
    return static_cast<uintptr_t>(pointer);
  }
}

ReturnStatus ValidateDeviceBuffer(void *buffer, size_t bytes,
                                  int current_device,
                                  MetaxAllocation *allocation) {
  int buffer_device = -1;
  const auto pointer = reinterpret_cast<mcDeviceptr_t>(buffer);
  auto status = CheckMaca(
      mcPointerGetAttribute(&buffer_device, mcPointerAttributeDevice, pointer),
      "mcPointerGetAttribute(device)");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  if (buffer_device != current_device) {
    std::cerr
        << "The RDMA buffer does not belong to the current MACA device.\n";
    return ReturnStatus::kInvalidArgument;
  }
  mcDeviceptr_t base{};
  size_t allocation_bytes = 0;
  status = CheckMaca(mcMemGetAddressRange(&base, &allocation_bytes, pointer),
                     "mcMemGetAddressRange");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  const auto address = reinterpret_cast<uintptr_t>(buffer);
  const auto start = PointerAddress(base);
  if (address < start || address - start > allocation_bytes ||
      bytes > allocation_bytes - (address - start)) {
    return ReturnStatus::kInvalidArgument;
  }
  allocation->base = reinterpret_cast<void *>(start);
  allocation->bytes = allocation_bytes;
  allocation->offset = address - start;
  return ReturnStatus::kSuccess;
}

bool ExperimentalDirectEnabled() {
  const char *value = std::getenv("INFINICCL_RDMA_METAX_EXPERIMENTAL");
  return value && std::strcmp(value, "1") == 0;
}

void QueryFlushPolicy(int device, bool *flush) {
  static std::once_flag warning_once;
  int ordering = 0;
  int options = 0;
  const auto ordering_status = mcDeviceGetAttribute(
      &ordering, mcDeviceAttributeGPUDirectRDMAWritesOrdering, device);
  const auto options_status = mcDeviceGetAttribute(
      &options, mcDeviceAttributeGPUDirectRDMAFlushWritesOptions, device);
  *flush = ordering_status == mcSuccess && options_status == mcSuccess &&
           ordering < mcFlushGPUDirectRDMAWritesToOwner &&
           (options & mcFlushGPUDirectRDMAWritesOptionHost);
  const bool native_ordering = ordering_status == mcSuccess &&
                               ordering >= mcFlushGPUDirectRDMAWritesToOwner;
  std::call_once(warning_once, [flush, native_ordering] {
    std::cerr << "Experimental MetaX DeviceDirect: DMA visibility is not yet "
                 "validated; no Host staging is used. Receive flush: "
              << (*flush            ? "enabled"
                  : native_ordering ? "native ordering reported"
                                    : "not reported, visibility unverified")
              << ".\n";
  });
}

ReturnStatus ExportDeviceMemory(void *buffer, size_t bytes,
                                RdmaMemoryRegion *region, MXuint64 *offset) {
  *offset = std::numeric_limits<MXuint64>::max();
  const auto status =
      mxkwExportDMABufHandle(buffer, bytes, &region->fd, offset);
  // The target MACA 3.2.1 probe returned zero on a successful export.
  if (static_cast<int>(status) != 0 || region->fd < 0 ||
      *offset > std::numeric_limits<MXuint64>::max() - bytes) {
    std::cerr << "`mxkwExportDMABufHandle` failed: status="
              << static_cast<int>(status) << ", fd=" << region->fd << ".\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus RegisterDeviceMemory(ibv_pd *pd, void *buffer, size_t bytes,
                                  const MetaxAllocation &allocation,
                                  RdmaMemoryRegion *region) {
  MXuint64 offset = 0;
  auto status =
      ExportDeviceMemory(allocation.base, allocation.bytes, region, &offset);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  if (allocation.offset > std::numeric_limits<MXuint64>::max() - offset) {
    return ReturnStatus::kInvalidArgument;
  }
  offset += allocation.offset;
  const auto iova = reinterpret_cast<uintptr_t>(buffer);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || offset % page_size != iova % page_size) {
    return ReturnStatus::kInvalidArgument;
  }
  region->mr = ibv_reg_dmabuf_mr(pd, offset, bytes, iova, region->fd,
                                 IBV_ACCESS_LOCAL_WRITE);
  if (!region->mr) {
    const int error = errno;
    std::cerr << "MetaX DMA-BUF MR registration failed: "
              << std::strerror(error) << " (" << error << ").\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

class MetaxMemoryAdapter : public RdmaMemoryAdapter {
 public:
  ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) override;

  ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                        RdmaMemoryRegion *region) override {
    return device_memory_
               ? RegisterDeviceMemory(pd, buffer, bytes, allocation_, region)
               : RegisterRdmaHostMemory(pd, buffer, bytes, region);
  }

  ReturnStatus Complete(bool receive) override {
    if (receive && device_memory_ && flush_) {
      return CheckMaca(mcDeviceFlushGPUDirectRDMAWrites(
                           mcFlushGPUDirectRDMAWritesTargetCurrentDevice,
                           mcFlushGPUDirectRDMAWritesToOwner),
                       "mcDeviceFlushGPUDirectRDMAWrites");
    }
    return ReturnStatus::kSuccess;
  }

 private:
  bool device_memory_ = false;

  bool flush_ = false;

  MetaxAllocation allocation_;
};

ReturnStatus MetaxMemoryAdapter::Prepare(void *buffer, size_t bytes,
                                         void *stream) {
  mcPointerAttribute_t attributes{};
  auto status = CheckMaca(mcPointerGetAttributes(&attributes, buffer),
                          "mcPointerGetAttributes");
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  device_memory_ = attributes.type == mcMemoryTypeDevice;
  if (!device_memory_ && attributes.type != mcMemoryTypeHost &&
      attributes.type != mcMemoryTypeUnregistered) {
    std::cerr << "Managed or array memory is not supported by MetaX RDMA.\n";
    return ReturnStatus::kNotSupported;
  }
  if (device_memory_) {
    if (!ExperimentalDirectEnabled()) {
      std::cerr << "MetaX DeviceDirect requires explicit experimental opt-in: "
                   "`INFINICCL_RDMA_METAX_EXPERIMENTAL=1`.\n";
      return ReturnStatus::kNotSupported;
    }
    int device = 0;
    status = CheckMaca(mcGetDevice(&device), "mcGetDevice");
    if (status == ReturnStatus::kSuccess) {
      status = ValidateDeviceBuffer(buffer, bytes, device, &allocation_);
    }
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    QueryFlushPolicy(device, &flush_);
  }
  return CheckMaca(mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream)),
                   "mcStreamSynchronize");
}

}  // namespace

ReturnStatus MemoryAdapterFactory<Device::Type::kMetax>::Create(
    std::unique_ptr<RdmaMemoryAdapter> *adapter) {
  if (!adapter) {
    return ReturnStatus::kInvalidArgument;
  }
  *adapter = std::make_unique<MetaxMemoryAdapter>();
  return ReturnStatus::kSuccess;
}

}  // namespace infini::ccl
