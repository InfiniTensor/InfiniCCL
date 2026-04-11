#ifndef INFINI_CCL_BACKEND_H_
#define INFINI_CCL_BACKEND_H_

#include <cstdint>

#include "traits.h"

namespace infini::ccl {

enum class BackendType : int8_t {
  kOmpi = 0,
  kGloo = 1,
  kNccl = 2,
  kMccl = 3,
  kRccl = 4,
  kCncl = 5,
  kHccl = 6,
  kCount
};

using AllBackendTypes =
    List<BackendType::kOmpi, BackendType::kGloo, BackendType::kNccl,
         BackendType::kMccl, BackendType::kRccl, BackendType::kCncl,
         BackendType::kHccl>;

template <typename Key, BackendType> struct BackendEnabled : std::false_type {};

/**
 * @brief Deferred computation of active backends for a specific operation Key.
 */
template <typename Key> struct ActiveBackendsImpl {
  struct Filter {
    template <auto kBackend>
    std::enable_if_t<
        BackendEnabled<Key, static_cast<BackendType>(kBackend)>::value>
    operator()(ValueTag<kBackend>) const {}
  };

  using type = typename FilterList<Filter, std::tuple<>, AllBackendTypes>::type;
};

template <typename Key>
using ActiveBackends = typename ActiveBackendsImpl<Key>::type;

/**
 * @brief Priority trait for backend selection.
 */
template <BackendType B> struct BackendPriority {
  static constexpr int value = 0;
};

template <> struct BackendPriority<BackendType::kOmpi> {
  static constexpr int value = 1;
};

template <> struct BackendPriority<BackendType::kNccl> {
  static constexpr int value = 10;
};

template <typename Key>
constexpr BackendType ListGetBestBackend(ActiveBackends<Key>) {
  static_assert(ListSize<ActiveBackends<Key>>::value > 0,
                "No backends enabled for this operation.");
  return ListGetMax<BackendPriority>(ActiveBackends<Key>{});
}

} // namespace infini::ccl

#endif // INFINI_CCL_BACKEND_H_
