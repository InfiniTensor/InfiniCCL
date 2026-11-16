#ifndef INFINI_CCL_DEVICES_MARS_DEVICE_H_
#define INFINI_CCL_DEVICES_MARS_DEVICE_H_

#include <hcr/hc_runtime.h>

#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kMars> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kMars>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }
  hcPointerAttribute_t attr{};
  if (hcPointerGetAttributes(&attr, ptr) != hcSuccess) {
    (void)hcGetLastError();
    return MemorySpace::kHost;
  }
  return attr.type == hcMemoryTypeDevice || attr.type == hcMemoryTypeManaged ||
                 attr.type == hcMemoryTypeArray
             ? MemorySpace::kDevice
             : MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MARS_DEVICE_H_
