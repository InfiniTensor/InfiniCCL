#ifndef INFINI_CCL_BACKENDS_RDMA_MEMORY_ADAPTOR_H_
#define INFINI_CCL_BACKENDS_RDMA_MEMORY_ADAPTOR_H_

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "device.h"
#include "return_status_impl.h"

namespace infini::ccl {

struct RdmaMemoryRegion {
  ~RdmaMemoryRegion();

  RdmaMemoryRegion() = default;

  RdmaMemoryRegion(const RdmaMemoryRegion &) = delete;

  RdmaMemoryRegion &operator=(const RdmaMemoryRegion &) = delete;

  RdmaMemoryRegion(RdmaMemoryRegion &&other) noexcept;

  RdmaMemoryRegion &operator=(RdmaMemoryRegion &&other) noexcept;

  ReturnStatus Reset();

  ibv_mr *mr = nullptr;

  int fd = -1;
};

// Identifies the memory allocation a buffer belongs to. Adapters fill it in
// `Prepare`; `tag` carries an allocation identity when the runtime provides
// one and stays zero when only the address range is known.
struct RdmaMemoryKey {
  uintptr_t base = 0;

  size_t bytes = 0;

  uint64_t tag = 0;
};

class RdmaMemoryAdapter {
 public:
  virtual ~RdmaMemoryAdapter() = default;

  virtual ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) = 0;

  virtual ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                                RdmaMemoryRegion *region) = 0;

  virtual ReturnStatus Complete(bool receive) = 0;

  virtual RdmaMemoryKey memory_key() const = 0;

  // Reports whether the buffer could take the RDMA path. This is a cheap,
  // side-effect free filter for the path decision; the authoritative checks
  // still run in `Prepare`, so a late failure there is a hard error once the
  // peer has confirmed the RDMA path.
  virtual bool CanRegister(const void *buffer, size_t bytes) const = 0;
};

struct MrCacheStats {
  uint64_t registrations = 0;

  uint64_t acquisitions = 0;

  size_t entries = 0;

  size_t retained_bytes = 0;
};

// Caches registered memory regions keyed by allocation range so repeated
// transfers on the same buffer reuse one registration instead of paying the
// pinning cost per message. Entries are validated against `tag` to detect
// freed-and-reused allocations and evicted least-recently-used first.
class MrCache {
 public:
  ReturnStatus Acquire(ibv_pd *pd, RdmaMemoryAdapter *adapter, void *buffer,
                       size_t bytes, ibv_mr **mr);

  // Releases every cached region. Unsafe release aborts the process, mirroring
  // `RdmaMemoryRegion` teardown semantics.
  void Reset();

  MrCacheStats Stats() const;

 private:
  struct Entry {
    RdmaMemoryRegion region;

    uint64_t tag = 0;

    uint64_t last_used = 0;
  };

  using Range = std::pair<uintptr_t, size_t>;

  ReturnStatus InsertLocked(const RdmaMemoryKey &key, ibv_pd *pd,
                            RdmaMemoryAdapter *adapter, void *buffer,
                            size_t bytes, ibv_mr **mr);

  void EvictOldestLocked();

  void EvictOverflowLocked();

  mutable std::mutex mutex_;

  std::map<Range, Entry> entries_;

  size_t total_bytes_ = 0;

  uint64_t clock_ = 0;

  uint64_t registrations_ = 0;

  uint64_t acquisitions_ = 0;
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
