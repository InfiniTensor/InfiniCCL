#ifndef INFINI_CCL_DEVICES_ASCEND_DEVICE_H_
#define INFINI_CCL_DEVICES_ASCEND_DEVICE_H_

#include <acl/acl.h>

#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kAscend> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kAscend>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  aclrtPtrAttributes attributes{};
  if (aclrtPointerGetAttributes(ptr, &attributes) != ACL_SUCCESS) {
    return MemorySpace::kHost;
  }

  return attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE
             ? MemorySpace::kDevice
             : MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ASCEND_DEVICE_H_
