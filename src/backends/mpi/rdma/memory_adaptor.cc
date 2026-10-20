#include "backends/mpi/rdma/memory_adaptor.h"

#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

#include "devices/cpu/rdma_memory.h"

#if defined(INFINICCL_WITH_METAX_RDMA)
#include "devices/metax/rdma_memory.h"
#endif

#if defined(INFINICCL_WITH_NVIDIA_RDMA)
#include "devices/nvidia/rdma_memory.h"
#endif

#include "dispatcher.h"
#include "logging.h"

namespace infini::ccl {

namespace {

// Cached registrations are capped so repeated transfers over distinct large
// buffers cannot pin unbounded memory; eviction is least-recently-used.
constexpr size_t kMrCacheLimitBytes = 256 << 20;

}  // namespace

RdmaMemoryRegion::RdmaMemoryRegion(RdmaMemoryRegion &&other) noexcept
    : mr(other.mr), fd(other.fd) {
  other.mr = nullptr;
  other.fd = -1;
}

RdmaMemoryRegion &RdmaMemoryRegion::operator=(
    RdmaMemoryRegion &&other) noexcept {
  if (this != &other) {
    mr = other.mr;
    fd = other.fd;
    other.mr = nullptr;
    other.fd = -1;
  }
  return *this;
}

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

ReturnStatus MrCache::Acquire(ibv_pd *pd, RdmaMemoryAdapter *adapter,
                              void *buffer, size_t bytes, ibv_mr **mr) {
  const RdmaMemoryKey key = adapter->memory_key();
  std::lock_guard<std::mutex> lock(mutex_);
  ++acquisitions_;
  auto cached = entries_.find(Range(key.base, key.bytes));
  if (cached != entries_.end()) {
    if (key.tag == 0 || cached->second.tag == key.tag) {
      cached->second.last_used = ++clock_;
      *mr = cached->second.region.mr;
      return ReturnStatus::kSuccess;
    }
    // The allocation behind this range was freed and its address reused.
    total_bytes_ -= cached->first.second;
    entries_.erase(cached);
  }
  auto status = InsertLocked(key, pd, adapter, buffer, bytes, mr);
  if (status != ReturnStatus::kSuccess && !entries_.empty()) {
    // Registration can fail once pinning resources are exhausted; free the
    // least recently used entry and retry once.
    EvictOldestLocked();
    status = InsertLocked(key, pd, adapter, buffer, bytes, mr);
  }
  if (status == ReturnStatus::kSuccess) {
    EvictOverflowLocked();
  }
  return status;
}

ReturnStatus MrCache::InsertLocked(const RdmaMemoryKey &key, ibv_pd *pd,
                                   RdmaMemoryAdapter *adapter, void *buffer,
                                   size_t bytes, ibv_mr **mr) {
  RdmaMemoryRegion region;
  auto status = adapter->Register(pd, buffer, bytes, &region);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  Entry entry;
  entry.tag = key.tag;
  entry.last_used = ++clock_;
  entry.region = std::move(region);
  total_bytes_ += key.bytes;
  ++registrations_;
  const auto inserted =
      entries_.emplace(Range(key.base, key.bytes), std::move(entry));
  *mr = inserted.first->second.region.mr;
  return ReturnStatus::kSuccess;
}

void MrCache::EvictOldestLocked() {
  auto oldest = entries_.begin();
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->second.last_used < oldest->second.last_used) {
      oldest = it;
    }
  }
  total_bytes_ -= oldest->first.second;
  entries_.erase(oldest);
}

void MrCache::EvictOverflowLocked() {
  while (total_bytes_ > kMrCacheLimitBytes && entries_.size() > 1) {
    EvictOldestLocked();
  }
}

void MrCache::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &item : entries_) {
    const auto status = item.second.region.Reset();
    if (status != ReturnStatus::kSuccess) {
      std::cerr << __FILE__ << ':' << __LINE__ << ' ' << __func__
                << ": Failed to release a cached RDMA memory region safely.\n";
      assert(status == ReturnStatus::kSuccess);
      std::abort();
    }
  }
  entries_.clear();
  total_bytes_ = 0;
}

MrCacheStats MrCache::Stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  MrCacheStats stats;
  stats.registrations = registrations_;
  stats.acquisitions = acquisitions_;
  stats.entries = entries_.size();
  stats.retained_bytes = total_bytes_;
  return stats;
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
