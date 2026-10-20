#include "backends/rdma/memory_adaptor.h"

#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "devices/cpu/rdma_memory.h"

#if defined(INFINICCL_WITH_METAX_RDMA)
#include "devices/metax/rdma_memory.h"
#endif

#include "dispatcher.h"
#include "logging.h"

namespace infini::ccl {

RdmaMemoryRegion::~RdmaMemoryRegion() {
  const auto status = Reset();
  if (status != ReturnStatus::kSuccess) {
    std::cerr << __FILE__ << ':' << __LINE__ << ' ' << __func__
              << ": Failed to release registered RDMA memory safely.\n";
    assert(status == ReturnStatus::kSuccess);
    std::abort();
  }
}

ReturnStatus RdmaMemoryRegion::Reset() {
  if (mr) {
    if (ibv_dereg_mr(mr) != 0) {
      std::cerr << __FILE__ << ':' << __LINE__ << ' ' << __func__
                << ": Failed to deregister RDMA memory safely.\n";
      assert(false && "RDMA memory deregistration failed.");
      std::abort();
    }
    mr = nullptr;
  }
  if (fd >= 0) {
    const int status = close(fd);
    fd = -1;
    if (status != 0) {
      LOG("Failed to close the DMA-BUF descriptor.");
      return ReturnStatus::kSystemError;
    }
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus RegisterRdmaHostMemory(ibv_pd *pd, void *buffer, size_t bytes,
                                    RdmaMemoryRegion *region) {
  if (!pd || !buffer || bytes == 0 || !region || region->mr ||
      region->fd >= 0) {
    return ReturnStatus::kInvalidArgument;
  }
  region->mr = ibv_reg_mr(pd, buffer, bytes, IBV_ACCESS_LOCAL_WRITE);
  if (!region->mr) {
    const int error = errno;
    std::cerr << "Host MR registration failed: " << std::strerror(error) << " ("
              << error << ").\n";
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus CreateRdmaMemoryAdapter(
    Device::Type device, std::unique_ptr<RdmaMemoryAdapter> *adapter) {
  if (!adapter) {
    return ReturnStatus::kInvalidArgument;
  }
  if (device < Device::Type::kCpu || device >= Device::Type::kCount) {
    return ReturnStatus::kNotSupported;
  }
  return DispatchFunc<AllDeviceTypes>(
      device,
      [&](auto tag) {
        constexpr Device::Type kDevice = decltype(tag)::value;
        return MemoryAdapterFactory<kDevice>::Create(adapter);
      },
      "CreateRdmaMemoryAdapter");
}

}  // namespace infini::ccl
