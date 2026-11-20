#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_DEVICE_SUPPORT_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_DEVICE_SUPPORT_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
#include <mpi-ext.h>
#include <mpi.h>
#endif

#include "device.h"

namespace infini::ccl {

enum class MpiDeviceSupport : std::uint8_t {
  kHostStaged = 0,
  kDeviceAware = 1,
};

enum class MpiDeviceAwareMode : std::uint8_t {
  kAuto,
  kOn,
  kOff,
};

template <Device::Type kDev>
struct MpiDeviceTraits {
  static constexpr bool kDeviceAwareCapable = false;

  static bool Query() { return false; }
};

template <>
struct MpiDeviceTraits<Device::Type::kNvidia> {
#if defined(INFINICCL_MPI_HAS_OMPI_CUDA_QUERY)
  static constexpr bool kDeviceAwareCapable = true;

  static bool Query() { return MPIX_Query_cuda_support() != 0; }
#else
  static constexpr bool kDeviceAwareCapable = false;

  static bool Query() { return false; }
#endif
};

template <>
struct MpiDeviceTraits<Device::Type::kMetax> {
  // Add the vendor query here after MetaX MPI exposes a stable interface.
  static constexpr bool kDeviceAwareCapable = false;

  static bool Query() { return false; }
};

inline MpiDeviceAwareMode MpiDeviceAwareModeFromEnv() {
  const char *mode = std::getenv("INFINICCL_MPI_DEVICE_AWARE");
  if (mode == nullptr || std::strcmp(mode, "auto") == 0) {
    return MpiDeviceAwareMode::kAuto;
  }
  if (std::strcmp(mode, "1") == 0) {
    return MpiDeviceAwareMode::kOn;
  }
  if (std::strcmp(mode, "0") == 0) {
    return MpiDeviceAwareMode::kOff;
  }
  return MpiDeviceAwareMode::kAuto;
}

template <Device::Type kDev>
MpiDeviceSupport ResolveMpiDeviceSupport() {
  const MpiDeviceAwareMode mode = MpiDeviceAwareModeFromEnv();
  if (mode == MpiDeviceAwareMode::kOff) {
    return MpiDeviceSupport::kHostStaged;
  }
  if (mode == MpiDeviceAwareMode::kOn) {
    return MpiDeviceSupport::kDeviceAware;
  }

  if constexpr (!MpiDeviceTraits<kDev>::kDeviceAwareCapable) {
    return MpiDeviceSupport::kHostStaged;
  } else {
    return MpiDeviceTraits<kDev>::Query() ? MpiDeviceSupport::kDeviceAware
                                          : MpiDeviceSupport::kHostStaged;
  }
}

inline bool MpiDeviceAwareUsable(MpiDeviceSupport support) {
  return support == MpiDeviceSupport::kDeviceAware;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_DEVICE_SUPPORT_H_
