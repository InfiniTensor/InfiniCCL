#ifndef INFINI_CCL_DEVICES_KUNLUN_DEVICE_H_
#define INFINI_CCL_DEVICES_KUNLUN_DEVICE_H_

// clang-format off
#include <xpu/runtime.h>
// clang-format on

#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kKunlun> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kKunlun>(const void* /*ptr*/) {
  // XRE exposes no pointer-attribute query; report unknown so callers cannot
  // act on a wrong host/device guess.
  return MemorySpace::kUnknown;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_KUNLUN_DEVICE_H_
