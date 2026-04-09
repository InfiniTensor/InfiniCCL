#ifndef INFINI_CCL_NVIDIA_DEVICE__H_
#define INFINI_CCL_NVIDIA_DEVICE__H_

#include "device.h"

namespace infini::ccl {

template <> struct DeviceEnabled<Device::Type::kNvidia> : std::true_type {};

} // namespace infini::ccl

#endif
