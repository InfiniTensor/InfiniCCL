#ifndef INFINI_CCL_BACKENDS_IBV_MEMORY_ADAPTOR_H_
#define INFINI_CCL_BACKENDS_IBV_MEMORY_ADAPTOR_H_

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

struct IbvMemoryRegion {
  ~IbvMemoryRegion();

  IbvMemoryRegion() = default;

  IbvMemoryRegion(const IbvMemoryRegion &) = delete;

  IbvMemoryRegion &operator=(const IbvMemoryRegion &) = delete;

  IbvMemoryRegion(IbvMemoryRegion &&other) noexcept;

  IbvMemoryRegion &operator=(IbvMemoryRegion &&other) noexcept;

  ReturnStatus Reset();

  ibv_mr *mr = nullptr;

  int fd = -1;
};

// Identifies the memory allocation a buffer belongs to. Adapters fill it in
// `Prepare`; `tag` carries an allocation identity when the runtime provides
// one and stays zero when only the address range is known.
struct IbvMemoryKey {
  uintptr_t base = 0;

  size_t bytes = 0;

  uint64_t tag = 0;
};

class IbvMemoryAdapter {
 public:
  virtual ~IbvMemoryAdapter() = default;

  virtual ReturnStatus Prepare(void *buffer, size_t bytes, void *stream) = 0;

  virtual ReturnStatus Register(ibv_pd *pd, void *buffer, size_t bytes,
                                IbvMemoryRegion *region) = 0;

  virtual ReturnStatus Complete(bool receive) = 0;

  virtual IbvMemoryKey memory_key() const = 0;
};

struct IbvMrCacheStats {
  uint64_t registrations = 0;

  uint64_t acquisitions = 0;

  size_t entries = 0;

  size_t retained_bytes = 0;
};

// Caches registered memory regions keyed by allocation range so repeated
// transfers on the same buffer reuse one registration instead of paying the
// pinning cost per message. Entries are validated against `tag` to detect
// freed-and-reused allocations and evicted least-recently-used first.
class IbvMrCache {
 public:
  ReturnStatus Acquire(ibv_pd *pd, IbvMemoryAdapter *adapter, void *buffer,
                       size_t bytes, ibv_mr **mr);

  // Releases every cached region. Unsafe release aborts the process,
  // mirroring `IbvMemoryRegion` teardown semantics.
  void Reset();

  IbvMrCacheStats Stats() const;

 private:
  struct Entry {
    IbvMemoryRegion region;

    uint64_t tag = 0;

    uint64_t last_used = 0;
  };

  using Range = std::pair<uintptr_t, size_t>;

  ReturnStatus InsertLocked(const IbvMemoryKey &key, ibv_pd *pd,
                            IbvMemoryAdapter *adapter, void *buffer,
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
  static ReturnStatus Create(std::unique_ptr<IbvMemoryAdapter> * /*adapter*/) {
    return ReturnStatus::kNotSupported;
  }
};

ReturnStatus RegisterIbvHostMemory(ibv_pd *pd, void *buffer, size_t bytes,
                                    IbvMemoryRegion *region);

ReturnStatus CreateIbvMemoryAdapter(
    Device::Type device, std::unique_ptr<IbvMemoryAdapter> *adapter);

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_MEMORY_ADAPTOR_H_
